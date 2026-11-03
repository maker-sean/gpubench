#!/usr/bin/env python3
"""
gpubench: a small, honest single-GPU benchmark with a configuration fingerprint.

Measures:
  1. Sustained vs peak dense compute (bf16 GEMM held for N minutes)
  2. Device memory bandwidth (large device-to-device copies)
  3. Inference throughput (vLLM, fixed prompts, fixed output length)

Every run captures a configuration fingerprint (driver, CUDA, power limit, clocks,
ECC, MIG, PCIe link, VBIOS, GPU UUID/serial) and continuous telemetry
(clocks, power, temperature, throttle reasons) so variance can be attributed.

Core question: how short can a defensible benchmark be, and does that depend on SKU?
Each GEMM run records 1-second throughput windows, so the analysis can ask what a
benchmark of any shorter length would have reported, and how far off it would be.

Usage:
  python gpubench.py run --provider runpod --price-per-hr 2.69
  python gpubench.py run --provider runpod --inference-only --label vllm-0.X   # software-stack variant
  python gpubench.py analyze --dir results --tol 2 --warmup 30
"""
import argparse
import csv
import glob
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


# ----------------------------------------------------------------------------- helpers

def sh(cmd, timeout=60):
    """Run a shell command, return stdout, empty string if it failed.

    nvidia-smi exits nonzero for fields a given driver or SKU does not support,
    so a nonzero exit means "not available here", not "the harness is broken".
    Without this check an unsupported field lands in the fingerprint as a line
    of English error text that looks like data.
    """
    try:
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return p.stdout.strip() if p.returncode == 0 else ""
    except Exception as e:  # noqa: BLE001
        return f"ERROR: {e}"


def smi_target(gpu):
    """The nvidia-smi selector for torch device index `gpu`.

    Containers on Runpod, Vast and Lambda routinely set CUDA_VISIBLE_DEVICES, so
    torch device 0 is often not nvidia-smi index 0. Addressing the card by UUID
    keeps the fingerprint and the telemetry on the GPU that ran the benchmark.
    """
    import torch
    cand = []
    try:
        u = str(torch.cuda.get_device_properties(gpu).uuid)
        cand.append(u if u.startswith("GPU-") else f"GPU-{u}")
    except Exception:  # noqa: BLE001
        pass
    vis = [x.strip() for x in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if x.strip()]
    if gpu < len(vis):
        cand.append(vis[gpu])
    cand.append(str(gpu))
    for t in cand:
        if sh(f"nvidia-smi -i {t} --query-gpu=uuid --format=csv,noheader"):
            return t
    return str(gpu)


def pct(xs, p):
    if not xs:
        return None
    s = sorted(xs)
    k = (len(s) - 1) * p / 100
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def cv(xs):
    if len(xs) < 2:
        return 0.0
    m = statistics.mean(xs)
    return statistics.stdev(xs) / m if m else 0.0


def to_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# ----------------------------------------------------------------------------- fingerprint

GPU_FIELDS = [
    "name", "uuid", "serial", "vbios_version", "driver_version", "pci.bus_id",
    "pcie.link.gen.max", "pcie.link.gen.current", "pcie.link.width.max",
    "pcie.link.width.current", "power.limit", "power.default_limit",
    "power.max_limit", "clocks.max.sm", "clocks.max.memory",
    "clocks.applications.graphics", "ecc.mode.current", "mig.mode.current",
    "memory.total", "compute_mode", "persistence_mode",
]


def fingerprint(gpu, target, provider, price, label, out_dir):
    import torch
    try:
        import vllm
        vllm_version = vllm.__version__
    except ImportError:
        vllm_version = None

    gpu_info = {}
    for f in GPU_FIELDS:  # query one at a time: one unsupported field would fail a batch query
        gpu_info[f] = sh(f"nvidia-smi -i {target} --query-gpu={f} --format=csv,noheader,nounits")

    cpu_model = ""
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass

    fp = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "price_per_hr": price,
        "label": label,
        "gpu_index": gpu,
        "smi_target": target,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": gpu_info,
        "software": {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "vllm": vllm_version,
            "cudnn": torch.backends.cudnn.version(),
            "python": platform.python_version(),
            "kernel": platform.release(),
        },
        "host": {
            "hostname": platform.node(),
            "cpu_model": cpu_model,
            "cpu_count": os.cpu_count(),
            "gpu_count": torch.cuda.device_count(),
        },
        "topology": sh("nvidia-smi topo -m"),
    }
    (out_dir / "nvidia_smi_q.txt").write_text(sh(f"nvidia-smi -q -i {target}"))
    (out_dir / "fingerprint.json").write_text(json.dumps(fp, indent=2))
    return fp


# ----------------------------------------------------------------------------- telemetry

THROTTLE_BITS = {
    "power_cap": 0x4,              # SW power cap
    "hw_slowdown": 0x8 | 0x80,     # HW slowdown, HW power brake
    "thermal": 0x20 | 0x40,        # SW thermal, HW thermal
}


class Telemetry:
    """Samples nvidia-smi in the background while a benchmark runs."""

    def __init__(self, path, gpu, interval_ms=500):
        helptext = sh("nvidia-smi --help-query-gpu")
        reason = ("clocks_event_reasons.active" if "clocks_event_reasons.active" in helptext
                  else "clocks_throttle_reasons.active")
        fields = f"timestamp,utilization.gpu,clocks.sm,clocks.mem,power.draw,temperature.gpu,{reason}"
        self.path = path
        self.cmd = ["nvidia-smi", "-i", str(gpu), f"--query-gpu={fields}",
                    "--format=csv,noheader,nounits", "-lms", str(interval_ms)]

    def __enter__(self):
        self.f = open(self.path, "w")
        self.f.write("timestamp,util,sm_mhz,mem_mhz,power_w,temp_c,reasons\n")
        self.f.flush()
        self.p = subprocess.Popen(self.cmd, stdout=self.f, stderr=subprocess.DEVNULL)
        return self

    def __exit__(self, *exc):
        self.p.terminate()
        self.p.wait()
        self.f.close()
        return False


def summarize_telemetry(path):
    sm, power, temp, flags = [], [], [], {k: 0 for k in THROTTLE_BITS}
    n = 0
    with open(path) as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 7:
                continue
            row = [c.strip() for c in row]
            n += 1
            for lst, val in ((sm, row[2]), (power, row[4]), (temp, row[5])):
                v = to_float(val)
                if v is not None:
                    lst.append(v)
            try:
                bits = int(row[6], 16)
            except ValueError:
                bits = 0
            for k, mask in THROTTLE_BITS.items():
                if bits & mask:
                    flags[k] += 1
    if n == 0:
        return {"samples": 0}
    return {
        "samples": n,
        "sm_mhz_mean": statistics.mean(sm) if sm else None,
        "sm_mhz_min": min(sm) if sm else None,
        "power_w_mean": statistics.mean(power) if power else None,
        "temp_c_max": max(temp) if temp else None,
        "throttle_fraction": {k: v / n for k, v in flags.items()},
    }


# ----------------------------------------------------------------------------- benchmarks

def bench_gemm(seconds, n, window=1.0):
    import torch
    a = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(n, n, device="cuda", dtype=torch.bfloat16)
    c = torch.empty(n, n, device="cuda", dtype=torch.bfloat16)
    flops = 2 * n ** 3
    for _ in range(20):
        torch.matmul(a, b, out=c)
    torch.cuda.synchronize()

    windows, durations = [], []
    start = time.perf_counter()
    while time.perf_counter() - start < seconds:
        t0 = time.perf_counter()
        iters = 0
        while True:
            for _ in range(10):
                torch.matmul(a, b, out=c)
            torch.cuda.synchronize()
            iters += 10
            if time.perf_counter() - t0 >= window:
                break
        dt = time.perf_counter() - t0
        windows.append(iters * flops / dt / 1e12)
        durations.append(dt)

    del a, b, c
    torch.cuda.empty_cache()
    second_half = windows[len(windows) // 2:]
    first = statistics.mean(windows[:5])
    sustained = statistics.median(second_half)
    return {
        "n": n, "dtype": "bf16", "seconds": seconds,
        "window_tflops": windows,
        "window_seconds": durations,  # measured: the time axis is data, not an assumption
        "elapsed_s": sum(durations),
        "peak_window_tflops": max(windows),
        "first_5s_tflops": first,
        "sustained_median_tflops": sustained,
        "sustained_p5_tflops": pct(second_half, 5),
        "sustained_cv": cv(second_half),
        "decay_first_to_sustained_pct": (first - sustained) / first * 100 if first else None,
    }


def bench_membw(gib, iters=50):
    import torch
    nbytes = int(gib * 2 ** 30)
    src = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(src)
    for _ in range(5):
        dst.copy_(src)
    torch.cuda.synchronize()
    bws = []
    for _ in range(iters):
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        dst.copy_(src)
        e.record()
        e.synchronize()
        bws.append(2 * nbytes / (s.elapsed_time(e) / 1000) / 1e9)  # read + write, GB/s
    del src, dst
    torch.cuda.empty_cache()
    return {"gib": gib, "iters": iters, "median_gbps": statistics.median(bws),
            "max_gbps": max(bws), "cv": cv(bws)}


class Inference:
    def __init__(self, model):
        from vllm import LLM
        self.model = model
        self.llm = LLM(model=model, dtype="bfloat16", gpu_memory_utilization=0.85, seed=0)

    def run(self, num_prompts, max_tokens):
        from vllm import SamplingParams
        prompts = [f"Explain, in technical detail, topic {i}: how a GPU schedules warps, "
                   f"manages memory, and why performance varies across machines."
                   for i in range(num_prompts)]
        sp = SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True)
        self.llm.generate(prompts[:8], sp)  # warmup
        t0 = time.perf_counter()
        outs = self.llm.generate(prompts, sp)
        dt = time.perf_counter() - t0
        out_toks = sum(len(o.outputs[0].token_ids) for o in outs)
        in_toks = sum(len(o.prompt_token_ids) for o in outs)
        return {"model": self.model, "num_prompts": num_prompts, "max_tokens": max_tokens,
                "seconds": dt, "output_tokens_per_s": out_toks / dt,
                "total_tokens_per_s": (in_toks + out_toks) / dt}


# ----------------------------------------------------------------------------- commands

def cmd_run(args):
    import torch
    assert torch.cuda.is_available(), "No CUDA device visible"
    torch.cuda.set_device(args.gpu)
    target = smi_target(args.gpu)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    uuid = sh(f"nvidia-smi -i {target} --query-gpu=uuid --format=csv,noheader")
    uuid8 = uuid[-8:] if uuid and not uuid.startswith("ERROR") else f"gpu{args.gpu}"
    out = Path(args.out) / args.provider / f"{stamp}_{uuid8}"
    out.mkdir(parents=True, exist_ok=True)

    fp = fingerprint(args.gpu, target, args.provider, args.price_per_hr, args.label, out)
    print(f"[gpubench] {fp['gpu']['name']} | driver {fp['gpu']['driver_version']} | "
          f"power limit {fp['gpu']['power.limit']} of {fp['gpu']['power.default_limit']} W | "
          f"smi target {target} -> {out}")

    results = {"repeats": []}
    for r in range(0 if args.inference_only else args.repeats):
        rep = {"repeat": r, "cold": r == 0}
        tpath = out / f"telemetry_gemm_r{r}.csv"
        print(f"[gpubench] repeat {r}: GEMM for {args.gemm_seconds}s")
        with Telemetry(tpath, target):
            rep["gemm"] = bench_gemm(args.gemm_seconds, args.gemm_n)
        rep["gemm"]["telemetry"] = summarize_telemetry(tpath)

        tpath = out / f"telemetry_membw_r{r}.csv"
        print(f"[gpubench] repeat {r}: memory bandwidth")
        with Telemetry(tpath, target):
            rep["membw"] = bench_membw(args.membw_gib)
        rep["membw"]["telemetry"] = summarize_telemetry(tpath)
        results["repeats"].append(rep)
        (out / "results.json").write_text(json.dumps(results, indent=2))

    if not args.skip_inference:
        # Inference runs in a fresh process, for two reasons. vLLM spawns its own
        # engine workers and can hang or abort when the parent already holds a CUDA
        # context, and a parent that just held tens of GB of GEMM tensors leaves the
        # caching allocator holding reservations, which silently changes what
        # gpu_memory_utilization=0.85 actually means.
        results["inference"] = run_inference_subprocess(args, out, target)

    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"[gpubench] done -> {out}")


def run_inference_subprocess(args, out, target):
    cmd = [sys.executable, os.path.abspath(__file__), "inference",
           "--gpu", str(args.gpu), "--smi-target", target, "--out-dir", str(out),
           "--model", args.model, "--prompts", str(args.prompts),
           "--max-tokens", str(args.max_tokens), "--repeats", str(args.repeats)]
    env = dict(os.environ, VLLM_WORKER_MULTIPROC_METHOD="spawn")
    print(f"[gpubench] inference in a fresh process: {args.model}")
    p = subprocess.run(cmd, env=env)
    path = out / "inference.json"
    if not path.exists():
        return {"skipped": f"inference subprocess exited {p.returncode} with no output"}
    return json.loads(path.read_text())


def cmd_inference(args):
    """The inference phase. `run` spawns this as its own process; see above."""
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        import torch
        torch.cuda.set_device(args.gpu)
        inf = Inference(args.model)
    except Exception as e:  # noqa: BLE001
        # A missing vLLM, a failed model download and an OOM all land here. Record
        # which one it was rather than losing the GEMM data to a traceback.
        (out / "inference.json").write_text(
            json.dumps({"skipped": f"{type(e).__name__}: {e}"}, indent=2))
        print(f"[gpubench] inference unavailable: {type(e).__name__}: {e}")
        return
    res_list = []
    for r in range(args.repeats):
        tpath = out / f"telemetry_infer_r{r}.csv"
        print(f"[gpubench] repeat {r}: inference")
        with Telemetry(tpath, args.smi_target):
            res = inf.run(args.prompts, args.max_tokens)
        res["repeat"] = r
        res["telemetry"] = summarize_telemetry(tpath)
        res_list.append(res)
        (out / "inference.json").write_text(json.dumps(res_list, indent=2))


def convergence(windows, seconds=None, tols=(1.0, 2.0, 5.0), warmup=0.0):
    """What would a benchmark of t seconds have reported, and how wrong would it be?

    reference  = median throughput over the final 25% of the run (steady state)
    estimate   = time-weighted mean throughput from `warmup` seconds up to t
    t          = total wall-clock length of the test, warm-up included, because a
                 test that throws away its first 30 s still occupies the GPU for them
    converge   = earliest t after which the estimate stays within tol% of reference

    warmup=0 is the naive estimator that averages from second one, which bakes the
    ramp into the number and so takes roughly (ramp length x drop / tol) seconds to
    wash out. A positive warmup discards the ramp instead. Reporting both is the
    only way to tell a property of the card from a property of the arithmetic.
    """
    n = len(windows or [])
    if n < 20:
        return None
    if not seconds or len(seconds) != n:
        seconds = [1.0] * n  # runs recorded before window_seconds existed
    end, acc = [], 0.0
    for s in seconds:
        acc += s
        end.append(acc)
    total = acc

    tail = [w for w, e in zip(windows, end) if e >= 0.75 * total]
    ref = statistics.median(tail or windows)
    if not ref:
        return None

    start_i = 0
    while start_i < n - 1 and end[start_i] - seconds[start_i] < warmup:
        start_i += 1

    ts, errors, num, den = [], [], 0.0, 0.0
    for i in range(start_i, n):
        num += windows[i] * seconds[i]
        den += seconds[i]
        ts.append(end[i])
        errors.append(abs(num / den - ref) / ref * 100)

    out = {"reference_tflops": ref, "warmup_s": warmup, "total_s": total,
           "t_s": ts, "errors_pct": errors}
    for tol in sorted({float(t) for t in tols}):
        t = ts[0] if ts else None
        for i in range(len(errors) - 1, -1, -1):
            if errors[i] > tol:
                t = ts[i + 1] if i + 1 < len(errors) else None  # None means never
                break
        out[f"t_converge_{tol:g}pct_s"] = t
    out["short_test_bias_pct_at_60s"] = short_test_bias(windows, seconds, end, ref, 60.0)
    return out


def short_test_bias(windows, seconds, end, ref, t):
    """Signed error of a t-second test that averages from second one.

    None when the run is shorter than t: a 30 s smoke test cannot report a 60 s bias,
    and the old code silently relabelled its own length as 60 s.
    """
    if not end or end[-1] < t or not ref:
        return None
    num = den = 0.0
    for w, s, e in zip(windows, seconds, end):
        if e > t:
            break
        num += w * s
        den += s
    return (num / den - ref) / ref * 100 if den else None


def spearman(xs, ys):
    if len(xs) < 3:
        return None

    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for rank, i in enumerate(order):
            r[i] = rank
        return r
    rx, ry = ranks(xs), ranks(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else None


def load_runs(root):
    runs = []
    for p in sorted(glob.glob(f"{root}/*/*/results.json")):
        d = Path(p).parent
        fpath = d / "fingerprint.json"
        if not fpath.exists():
            print(f"[gpubench] skipping {d}: no fingerprint.json")
            continue
        runs.append({"dir": d, "fp": json.loads(fpath.read_text()),
                     "res": json.loads(Path(p).read_text())})
    return runs


def fmt(v, d=1):
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{d}f}"
    return str(v)


def med(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def worst(xs):
    if not xs:
        return "n/a"
    if any(x is None for x in xs):
        return "never"
    return fmt(max(xs))


def length_table(rows, lines):
    lines += ["| SKU | repeats | sustained TFLOPS | cold: median s | cold: worst s | "
              "warm: median s | warm: worst s | never converged | 60s bias % | median $/hr | "
              "cold converged test $ |", "|" + "---|" * 11]
    for sku in sorted({r["sku"] for r in rows}):
        rs = [r for r in rows if r["sku"] == sku]
        cold = [r["t_tol"] for r in rs if r["cold"]]
        warm = [r["t_tol"] for r in rs if not r["cold"]]
        costs = [r["price"] * r["t_tol"] / 3600 for r in rs
                 if r["cold"] and r["price"] and r["t_tol"]]
        lines.append(
            f"| {sku} | {len(rs)} | {fmt(med([r['sustained'] for r in rs]))} | "
            f"{fmt(med(cold))} | {worst(cold)} | {fmt(med(warm))} | {worst(warm)} | "
            f"{sum(1 for r in rs if r['t_tol'] is None)} of {len(rs)} | "
            f"{fmt(med([r['bias60'] for r in rs]))} | {fmt(med([r['price'] for r in rs]), 2)} | "
            f"{fmt(med(costs), 3) if costs else 'n/a'} |")


def fleet_table(rows, fleet, lines):
    lines += [f"| SKU | cold s to tol | GPU-hours for {fleet:,} | $ for {fleet:,} | "
              "fixed 60 s GPU-hours | fixed 60 s $ | what a fixed 60 s test does |",
              "|" + "---|" * 7]
    for sku in sorted({r["sku"] for r in rows}):
        rs = [r for r in rows if r["sku"] == sku]
        cold = [r["t_tol"] for r in rs if r["cold"]]
        t = med(cold)
        price = med([r["price"] for r in rs])
        gh = fleet * t / 3600 if t else None
        gh60 = fleet * 60 / 3600
        n_never = sum(1 for x in cold if x is None)
        if t is None:
            verdict = "no cold run converged: no defensible length exists in this data"
        elif n_never:
            verdict = (f"{n_never} of {len(cold)} cold runs never converged, "
                       f"so {t:.0f} s is a lower bound")
        elif t <= 60:
            verdict = f"converges, pays {60 - t:.0f} s/GPU it does not need"
        else:
            verdict = f"misrates: {t - 60:.0f} s/GPU short of converged"
        lines.append(f"| {sku} | {fmt(t)} | {fmt(gh)} | "
                     f"{fmt(gh * price if gh and price else None, 0)} | {fmt(gh60)} | "
                     f"{fmt(gh60 * price if price else None, 0)} | {verdict} |")


def cmd_analyze(args):
    runs = load_runs(args.dir)
    if not runs:
        print("No runs found.")
        return
    tols = (1.0, 2.0, 5.0, float(args.tol))
    key = f"t_converge_{args.tol:g}pct_s"
    estimators = [("from_start", 0.0), (f"warmup{args.warmup:g}s", float(args.warmup))]
    names = [n for n, _ in estimators]
    per_run = {n: [] for n in names}
    curve_rows = []

    for run in runs:
        fp, res = run["fp"], run["res"]
        sku = fp["gpu"]["name"]
        for rep in res.get("repeats", []):
            g = rep.get("gemm") or {}
            w, s = g.get("window_tflops"), g.get("window_seconds")
            if not w:
                continue
            tel = g.get("telemetry", {})
            for name, warm in estimators:
                c = convergence(w, s, tols=tols, warmup=warm)
                if not c:
                    continue
                for t, e in zip(c["t_s"], c["errors_pct"]):
                    curve_rows.append([name, sku, fp["provider"], run["dir"].name, rep["repeat"],
                                       "cold" if rep.get("cold") else "warm",
                                       round(t, 3), round(e, 4)])
                per_run[name].append({
                    "estimator": name, "sku": sku, "provider": fp["provider"],
                    "run": run["dir"].name, "uuid": fp["gpu"].get("uuid"),
                    "repeat": rep["repeat"], "cold": rep.get("cold", False),
                    "price": fp.get("price_per_hr"),
                    "power_limit": fp["gpu"].get("power.limit"),
                    "driver": fp["gpu"].get("driver_version"),
                    "sustained": c["reference_tflops"], "run_length_s": c["total_s"],
                    "t1": c["t_converge_1pct_s"], "t2": c["t_converge_2pct_s"],
                    "t5": c["t_converge_5pct_s"], "t_tol": c[key],
                    "bias60": c["short_test_bias_pct_at_60s"],
                    "power_cap_frac": tel.get("throttle_fraction", {}).get("power_cap"),
                    "thermal_frac": tel.get("throttle_fraction", {}).get("thermal"),
                    "membw": (rep.get("membw") or {}).get("median_gbps"),
                })

    lines = ["# gpubench summary", "",
             f"Tolerance {args.tol:g}%. Benchmark length is total wall time on the GPU, "
             f"warm-up included. Fleet figures assume {args.fleet:,} GPUs.", ""]

    # Headline: required test length by SKU, cold vs warm, under both estimators
    lines += [f"## Benchmark length needed to land within {args.tol:g}% of steady state", ""]
    for name, warm in estimators:
        lines += [f"### Estimator `{name}`: " + ("average from second one"
                  if not warm else f"discard the first {warm:g} s"), ""]
        length_table(per_run[name], lines)
        lines.append("")
    lines += [f"A warm-up estimator cannot report a test shorter than its own warm-up, so "
              f"`{names[1]}` floors at about {args.warmup:g} s. Where a SKU shows that floor, the "
              f"estimator is setting the test length, not the card.", "",
              "Positive 60s bias means a one-minute test overstates steady-state performance.",
              "60s bias is the from_start estimator by definition: it is what a naive "
              "one-minute test reports. It is blank for runs shorter than 60 s.", ""]

    # The estimator decision, side by side
    lines += ["## Which estimator", "",
              "| SKU | state | " + " | ".join(f"{n} median s" for n in names) + " | "
              + " | ".join(f"{n} never" for n in names) + " |",
              "|" + "---|" * (2 + 2 * len(names))]
    for sku in sorted({r["sku"] for r in per_run[names[0]]}):
        for state in ("cold", "warm"):
            sel = {n: [r["t_tol"] for r in per_run[n]
                       if r["sku"] == sku and r["cold"] == (state == "cold")] for n in names}
            if not any(sel.values()):
                continue
            lines.append(f"| {sku} | {state} | "
                         + " | ".join(fmt(med(sel[n])) for n in names) + " | "
                         + " | ".join(str(sum(1 for x in sel[n] if x is None)) for n in names)
                         + " |")
    lines.append("")

    # Fleet-scale cost
    lines += [f"## Cost to certify {args.fleet:,} GPUs at {args.tol:g}%", ""]
    for name, warm in estimators:
        lines += [f"### Estimator `{name}`", ""]
        fleet_table(per_run[name], args.fleet, lines)
        lines.append("")
    lines += ["Cold time-to-converge is also the idle time each seller gives up per card.",
              "The cold column is a median over the cold runs that converged; runs that never "
              "converged are counted in the verdict, not averaged into the number.", ""]

    # Per-run detail
    lines += ["## Per run", "",
              "| SKU | provider | run | rep | state | power W | driver | sustained | run len s | "
              + " | ".join(f"t {n}" for n in names) + " | t1% | t5% | power-cap frac | "
              "thermal frac | mem GB/s |",
              "|" + "---|" * (14 + len(names))]
    other = {(r["run"], r["repeat"]): r for r in per_run[names[1]]}
    for r in per_run[names[0]]:
        o = other.get((r["run"], r["repeat"]), {})
        lines.append(f"| {r['sku']} | {r['provider']} | {r['run']} | {r['repeat']} | "
                     f"{'cold' if r['cold'] else 'warm'} | {r['power_limit']} | {r['driver']} | "
                     f"{fmt(r['sustained'])} | {fmt(r['run_length_s'], 0)} | "
                     f"{fmt(r['t_tol'])} | {fmt(o.get('t_tol'))} | "
                     f"{fmt(r['t1'])} | {fmt(r['t5'])} | "
                     f"{fmt(r['power_cap_frac'], 2)} | {fmt(r['thermal_frac'], 2)} | "
                     f"{fmt(r['membw'], 0)} |")

    # Does the synthetic score predict the workload? (within SKU, across instances)
    lines += ["", "## Does the compute score predict inference throughput?", ""]
    inst = []
    for run in runs:
        inf = run["res"].get("inference")
        reps = run["res"].get("repeats", [])
        if isinstance(inf, list) and inf and reps:
            sus = []
            for rep in reps:
                g = rep.get("gemm") or {}
                c = convergence(g.get("window_tflops"), g.get("window_seconds"), tols=tols)
                if c:
                    sus.append(c["reference_tflops"])
            tok = med([i["output_tokens_per_s"] for i in inf])
            price = run["fp"].get("price_per_hr")
            inst.append({"sku": run["fp"]["gpu"]["name"], "provider": run["fp"]["provider"],
                         "run": run["dir"].name, "sus": med(sus), "tok": tok, "price": price,
                         "usd_m": price / (tok * 3600) * 1e6 if price and tok else None})
    for sku in sorted({i["sku"] for i in inst}):
        xs = [i for i in inst if i["sku"] == sku and i["sus"] and i["tok"]]
        rho = spearman([i["sus"] for i in xs], [i["tok"] for i in xs])
        lines.append(f"- {sku}: {len(xs)} instances, rank correlation {fmt(rho, 2)}"
                     f"{' (needs 3+ instances; 2 per SKU cannot produce one)' if rho is None else ''}")
    if inst:
        lines += ["", "| SKU | provider | run | sustained TFLOPS | out tok/s | $/hr | $/M out tok |",
                  "|---|---|---|---|---|---|---|"]
        for i in inst:
            lines.append(f"| {i['sku']} | {i['provider']} | {i['run']} | {fmt(i['sus'])} | "
                         f"{fmt(i['tok'], 0)} | {fmt(i['price'], 2)} | {fmt(i['usd_m'], 3)} |")

    # Software vs hardware: same physical GPU, different software labels
    by_uuid = {}
    for run in runs:
        inf = run["res"].get("inference")
        if isinstance(inf, list) and inf:
            u = run["fp"]["gpu"].get("uuid")
            by_uuid.setdefault(u, []).append((run["fp"].get("label", "default"),
                                              run["fp"]["software"].get("vllm"),
                                              run["fp"]["software"].get("torch"),
                                              med([i["output_tokens_per_s"] for i in inf])))
    sw = {u: v for u, v in by_uuid.items() if len({x[0] for x in v}) > 1}
    lines += ["", "## Same GPU, different software stack", ""]
    if not sw:
        lines.append("No GPU was run under more than one --label yet.")
    for u, v in sw.items():
        base = v[0][3]
        lines.append(f"- GPU {u[-8:] if u else '?'}:")
        for label, vv, tv, tok in v:
            lines.append(f"  - {label} (vllm {vv}, torch {tv}): {fmt(tok, 0)} tok/s, "
                         f"{fmt((tok - base) / base * 100 if base and tok else None)}% vs first")

    # Anything that would make a number untrustworthy
    lines += ["", "## Data quality flags", ""]
    flags = []
    for r in per_run[names[0]]:
        tag = f"{r['sku']} {r['run']} rep {r['repeat']}"
        if r["run_length_s"] and r["run_length_s"] < 300:
            flags.append(f"- {tag}: run only {r['run_length_s']:.0f} s long, "
                         "too short to establish steady state")
        if r["t_tol"] is None:
            flags.append(f"- {tag}: never converged within the run, so its steady state "
                         "is an assumption, not a measurement")
        if r["bias60"] is None:
            flags.append(f"- {tag}: shorter than 60 s, no 60s bias available")
    for run in runs:
        inf = run["res"].get("inference")
        if isinstance(inf, dict) and inf.get("skipped"):
            flags.append(f"- {run['dir'].name}: inference skipped ({inf['skipped']})")
        for rep in run["res"].get("repeats", []):
            n = ((rep.get("gemm") or {}).get("telemetry") or {}).get("samples", 0)
            if not n:
                flags.append(f"- {run['dir'].name} rep {rep.get('repeat')}: GEMM telemetry empty, "
                             "throttle attribution unavailable")
    lines += sorted(set(flags)) or ["None."]

    text = "\n".join(lines)
    Path(args.dir, "summary.md").write_text(text)
    Path(args.dir, "summary.json").write_text(json.dumps(
        {"tol": args.tol, "warmup": args.warmup, "fleet": args.fleet,
         "estimators": names, "runs": per_run, "instances": inst}, indent=2))
    with open(Path(args.dir, "convergence.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["estimator", "sku", "provider", "run", "repeat", "state",
                    "seconds", "error_pct"])
        w.writerows(curve_rows)
    print(text)
    print(f"\nWrote summary.md, summary.json, convergence.csv to {args.dir}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="benchmark this machine")
    r.add_argument("--provider", required=True, help="label, e.g. runpod, lambda, vast")
    r.add_argument("--price-per-hr", type=float, default=None, help="what you paid per GPU-hour")
    r.add_argument("--gpu", type=int, default=0)
    r.add_argument("--repeats", type=int, default=3)
    r.add_argument("--gemm-seconds", type=int, default=1200,
                   help="long enough to reach steady state on slow-settling SKUs")
    r.add_argument("--gemm-n", type=int, default=8192)
    r.add_argument("--membw-gib", type=float, default=2.0)
    r.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    r.add_argument("--prompts", type=int, default=256)
    r.add_argument("--max-tokens", type=int, default=256)
    r.add_argument("--skip-inference", action="store_true")
    r.add_argument("--inference-only", action="store_true",
                   help="skip GEMM and bandwidth; use with --label to compare software stacks on one box")
    r.add_argument("--label", default="default", help="software-stack label, e.g. vllm-0.8")
    r.add_argument("--out", default="results")
    r.set_defaults(func=cmd_run)

    i = sub.add_parser("inference", help="internal: the inference phase, spawned by run")
    i.add_argument("--gpu", type=int, default=0)
    i.add_argument("--smi-target", default="0")
    i.add_argument("--out-dir", required=True)
    i.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    i.add_argument("--prompts", type=int, default=256)
    i.add_argument("--max-tokens", type=int, default=256)
    i.add_argument("--repeats", type=int, default=3)
    i.set_defaults(func=cmd_inference)

    a = sub.add_parser("analyze", help="summarize all runs")
    a.add_argument("--dir", default="results")
    a.add_argument("--tol", type=float, default=2.0,
                   help="error tolerance in %% that defines a converged benchmark")
    a.add_argument("--warmup", type=float, default=30.0,
                   help="seconds the second estimator discards as warm-up")
    a.add_argument("--fleet", type=int, default=10000,
                   help="fleet size for the cost-to-certify figures")
    a.set_defaults(func=cmd_analyze)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
