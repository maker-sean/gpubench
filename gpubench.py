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


STAMP_FILE = ".gpubench_instance_start"


def resolve_instance_start(arg):
    """Unix time at which this GPU started costing money.

    Order of preference: the --instance-start you pass (when you clicked deploy, the
    only truthful answer), then the stamp setup.sh wrote when it first ran on the box
    (a lower bound: it misses provisioning, boot and image pull), then nothing.
    """
    if arg:
        try:
            return float(arg)
        except ValueError:
            return datetime.fromisoformat(arg.replace("Z", "+00:00")).timestamp()
    for q in (Path(STAMP_FILE), Path.home() / STAMP_FILE):
        if q.exists():
            try:
                return float(q.read_text().strip())
            except (ValueError, OSError):
                pass
    return None


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
    # Clock against temperature is the discriminator the throttle flags miss. A card held
    # at its power limit sheds clock as it warms, because hotter silicon leaks more current
    # and the same watts buy fewer MHz. Nothing sets a thermal flag while that happens: the
    # reported reason stays SwPowerCap throughout. Branching on thermal_fraction alone would
    # call such a card thermally unaffected.
    q = max(len(sm) // 4, 1)
    q1, q4 = (statistics.mean(sm[:q]), statistics.mean(sm[-q:])) if sm else (None, None)
    corr = None
    if len(sm) > 2 and len(temp) == len(sm):
        ms, mt = statistics.mean(sm), statistics.mean(temp)
        den = (sum((a - ms) ** 2 for a in sm) * sum((b - mt) ** 2 for b in temp)) ** 0.5
        if den:
            corr = sum((a - ms) * (b - mt) for a, b in zip(sm, temp)) / den
    return {
        "samples": n,
        "sm_mhz_q1_mean": q1,
        "sm_mhz_q4_mean": q4,
        "sm_decline_pct": (q1 - q4) / q1 * 100 if q1 else None,
        "clock_temp_corr": corr,
        "sm_mhz_mean": statistics.mean(sm) if sm else None,
        "sm_mhz_min": min(sm) if sm else None,
        "power_w_mean": statistics.mean(power) if power else None,
        "temp_c_max": max(temp) if temp else None,
        "temp_c_first": temp[0] if temp else None,
        "temp_c_last": temp[-1] if temp else None,
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
    t_proc = time.time()
    import torch
    assert torch.cuda.is_available(), "No CUDA device visible"
    torch.cuda.set_device(args.gpu)
    target = smi_target(args.gpu)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    uuid = sh(f"nvidia-smi -i {target} --query-gpu=uuid --format=csv,noheader")
    uuid8 = uuid[-8:] if uuid and not uuid.startswith("ERROR") else f"gpu{args.gpu}"
    out = Path(args.out) / args.provider / f"{stamp}_{uuid8}"
    out.mkdir(parents=True, exist_ok=True)

    timings = {"instance_start": resolve_instance_start(args.instance_start),
               "process_start": t_proc, "first_window": None}
    fp = fingerprint(args.gpu, target, args.provider, args.price_per_hr, args.label, out)
    timings["fingerprint_done"] = time.time()
    print(f"[gpubench] {fp['gpu']['name']} | driver {fp['gpu']['driver_version']} | "
          f"power limit {fp['gpu']['power.limit']} of {fp['gpu']['power.default_limit']} W | "
          f"smi target {target} -> {out}")

    if args.fingerprint_only:
        # Census mode: record what this host actually gave us, then get off the clock.
        # Power limit is not published by any provider before rental, so the only way to
        # survey it is to rent briefly and read it.
        (out / "results.json").write_text(json.dumps({"repeats": [], "timings": timings,
                                                      "census": True}, indent=2))
        print(f"[gpubench] census: {fp['gpu']['name']} | power {fp['gpu']['power.limit']} of "
              f"{fp['gpu']['power.default_limit']} W | clocks.max.sm {fp['gpu']['clocks.max.sm']} "
              f"| driver {fp['gpu']['driver_version']} -> {out}")
        return

    results = {"repeats": [], "timings": timings}

    def one_repeat(r, state):
        rep = {"repeat": r, "cold": state == "cold", "cooled": state == "recooled",
               "state": state}
        tpath = out / f"telemetry_gemm_r{r}.csv"
        print(f"[gpubench] repeat {r} ({state}): GEMM for {args.gemm_seconds}s")
        t0 = time.time()
        with Telemetry(tpath, target):
            rep["gemm"] = bench_gemm(args.gemm_seconds, args.gemm_n)
        rep["gemm"]["telemetry"] = summarize_telemetry(tpath)
        rep["gemm"]["started_at"] = t0
        if timings.get("first_window") is None:
            # The first second of real measurement. Everything before this is overhead.
            timings["first_window"] = t0 + (rep["gemm"]["window_seconds"] or [0])[0]

        tpath = out / f"telemetry_membw_r{r}.csv"
        print(f"[gpubench] repeat {r}: memory bandwidth")
        with Telemetry(tpath, target):
            rep["membw"] = bench_membw(args.membw_gib)
        rep["membw"]["telemetry"] = summarize_telemetry(tpath)
        results["repeats"].append(rep)
        (out / "results.json").write_text(json.dumps(results, indent=2))

    for r in range(0 if args.inference_only else args.repeats):
        one_repeat(r, "cold" if r == 0 else "warm")

    if args.cooldown and not args.inference_only:
        # Repeat 0 is cold and first-ever, so it absorbs thermal state AND one-time
        # software cost (cuBLAS autotune, first-touch allocation). Idling the card and
        # running once more separates them: if this matches repeat 0 the penalty is
        # thermal, if it matches the warm repeats it was software and happens once.
        print(f"[gpubench] cooling down {args.cooldown} min, telemetry running")
        tpath = out / "telemetry_idle.csv"
        with Telemetry(tpath, target):
            time.sleep(args.cooldown * 60)
        results["cooldown"] = {"minutes": args.cooldown,
                               "telemetry": summarize_telemetry(tpath)}
        one_repeat(args.repeats, "recooled")

    timings["gemm_done"] = time.time()

    if not args.skip_inference:
        # Inference runs in a fresh process, for two reasons. vLLM spawns its own
        # engine workers and can hang or abort when the parent already holds a CUDA
        # context, and a parent that just held tens of GB of GEMM tensors leaves the
        # caching allocator holding reservations, which silently changes what
        # gpu_memory_utilization=0.85 actually means.
        results["inference"] = run_inference_subprocess(args, out, target)

    timings["run_end"] = time.time()
    if isinstance(results.get("inference"), list) and results["inference"]:
        timings["model_ready"] = results["inference"][0].get("model_ready")
    (out / "results.json").write_text(json.dumps(results, indent=2))
    o = overhead(timings)
    if o is not None:
        print(f"[gpubench] overhead before the first measured second: {o:.0f}s")
    else:
        print("[gpubench] no instance start recorded, overhead unknown. "
              "Pass --instance-start next time or run setup.sh first.")
    print(f"[gpubench] done -> {out}")


def overhead(t):
    """Seconds of billed GPU time before the first measured second. None if unknown."""
    if not t:
        return None
    a, b = t.get("instance_start"), t.get("first_window")
    return b - a if a and b else None


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
        t_load = time.time()
        inf = Inference(args.model)
        model_ready = time.time()
        print(f"[gpubench] model ready in {model_ready - t_load:.0f}s "
              "(download plus load, billed GPU time)")
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
        res["model_load_s"] = model_ready - t_load
        res["model_ready"] = model_ready
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


T_CRIT_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365,
             8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 15: 2.131, 20: 2.086}


def t_crit(df):
    for k in sorted(T_CRIT_95):
        if df <= k:
            return T_CRIT_95[k]
    return 1.96


def _trend(ts, vals):
    """OLS slope of vals on ts, its standard error, and |t| statistic.

    Returned as a hypothesis test, not an extrapolation. "I cannot detect a trend"
    is a claim the data supports. "I project 1.8% of drift over the next ten minutes"
    is a noisy slope multiplied by a horizon, which inflates with the promise rather
    than with anything the card did. The earlier rule made that mistake.
    """
    n = len(ts)
    if n < 3:
        return 0.0, 0.0, 0.0
    mt, mv = statistics.mean(ts), statistics.mean(vals)
    den = sum((t - mt) ** 2 for t in ts)
    if not den:
        return 0.0, 0.0, 0.0
    slope = sum((t - mt) * (v - mv) for t, v in zip(ts, vals)) / den
    resid = [v - (mv + slope * (t - mt)) for t, v in zip(ts, vals)]
    sse = sum(r * r for r in resid)
    se = (sse / (n - 2) / den) ** 0.5 if n > 2 and sse > 0 else 0.0
    return slope, se, (abs(slope) / se if se else 0.0)


def sequential_stop(windows, seconds=None, tol=2.0, window_s=60.0, batches=6, confirm=2):
    """A stopping rule with two checks and no extrapolation.

    At each second, looking only at the last `window_s` seconds of the past:
      precise  the 95% confidence interval on the mean is inside tol%. Built from
               batch means, because consecutive seconds on one GPU are correlated
               and a naive standard error would claim confidence the data lacks.
      settled  the trend across that window is not statistically distinguishable
               from zero at 95%.

    Both must hold for `confirm` consecutive checks. Neither asks how long the
    answer will remain true: you cannot certify stability over a horizon longer
    than you observed, and pretending otherwise just multiplies noise by the
    promise. Durability is a repeat-measurement question, not a longer-run one.
    """
    n = len(windows or [])
    if n < 20:
        return None
    if not seconds or len(seconds) != n:
        seconds = [1.0] * n
    end, acc = [], 0.0
    for x in seconds:
        acc += x
        end.append(acc)
    tail = [w for w, e in zip(windows, end) if e >= 0.75 * acc]
    ref = statistics.median(tail or windows)
    if not ref:
        return None
    base = {"reference_tflops": ref, "window_s": window_s, "batches": batches}
    tc, streak = t_crit(batches - 1), 0

    for i in range(n):
        if end[i] < window_s:
            continue
        idx = [j for j in range(i + 1) if end[j] > end[i] - window_s]
        if len(idx) < batches * 3:
            continue
        vals = [windows[j] for j in idx]
        ts = [end[j] for j in idx]
        m = statistics.mean(vals)
        if not m:
            continue
        b = len(vals) // batches
        bm = [statistics.mean(vals[k * b:(k + 1) * b]) for k in range(batches)]
        bt = [statistics.mean(ts[k * b:(k + 1) * b]) for k in range(batches)]
        hw_pct = tc * statistics.stdev(bm) / (batches ** 0.5) / m * 100
        slope, _se, tstat = _trend(bt, bm)

        streak = streak + 1 if (hw_pct <= tol and tstat < t_crit(batches - 2)) else 0
        if streak >= confirm:
            err = abs(m - ref) / ref * 100
            return dict(base, stopped=True, t_stop_s=end[i], estimate_tflops=m,
                        ci_halfwidth_pct=hw_pct, trend_t=tstat,
                        trend_pct_per_min=slope * 60 / m * 100,
                        realized_error_pct=err, within_tol=err <= tol)
    return dict(base, stopped=False, t_stop_s=None, estimate_tflops=None,
                ci_halfwidth_pct=None, trend_t=None, trend_pct_per_min=None,
                realized_error_pct=None, within_tol=False)


def settled_check(windows, seconds=None, telemetry=None, batches=10):
    """Was this run settled enough to be scored at all? Judged after the fact.

    Two independent readings, so neither has to carry the verdict alone: a trend
    test on the second half of the throughput series, and clock and power stability
    from telemetry. A run that fails this should not produce a rating whatever the
    stopping rule said.
    """
    n = len(windows or [])
    if n < 40:
        return None
    if not seconds or len(seconds) != n:
        seconds = [1.0] * n
    end, acc = [], 0.0
    for x in seconds:
        acc += x
        end.append(acc)
    half = [(e, w) for w, e in zip(windows, end) if e >= 0.5 * acc]
    vals = [w for _e, w in half]
    ts = [e for e, _w in half]
    m = statistics.mean(vals)
    b = max(len(vals) // batches, 1)
    bm = [statistics.mean(vals[k * b:(k + 1) * b]) for k in range(batches)]
    bt = [statistics.mean(ts[k * b:(k + 1) * b]) for k in range(batches)]
    slope, _se, tstat = _trend(bt, bm)
    pct_per_min = slope * 60 / m * 100 if m else None
    flat = tstat < t_crit(batches - 2)

    tel = telemetry or {}
    thr = tel.get("throttle_fraction", {}) or {}
    return {
        "second_half_trend_pct_per_min": pct_per_min,
        "trend_t": tstat,
        "throughput_flat": flat,
        "sm_mhz_mean": tel.get("sm_mhz_mean"),
        "sm_mhz_min": tel.get("sm_mhz_min"),
        "power_w_mean": tel.get("power_w_mean"),
        "temp_c_max": tel.get("temp_c_max"),
        "power_cap_frac": thr.get("power_cap"),
        "thermal_frac": thr.get("thermal"),
        "sm_decline_pct": tel.get("sm_decline_pct"),
        "clock_temp_corr": tel.get("clock_temp_corr"),
        "limit": limit_reading(tel),
        "verdict": "settled" if flat else "still moving: do not rate this run",
    }


def limit_reading(tel):
    """Which limit actually shaped this run, read from clocks rather than flags."""
    thr = (tel or {}).get("throttle_fraction", {}) or {}
    dec, corr = tel.get("sm_decline_pct"), tel.get("clock_temp_corr")
    pw, th = thr.get("power_cap"), thr.get("thermal")
    if dec is None:
        return "unknown: no telemetry"
    if th and th > 0.05:
        return "thermal throttling, flagged"
    if dec > 2 and corr is not None and corr < -0.5:
        return ("thermally mediated under a power cap: clock fell "
                f"{dec:.1f}% as the card warmed, correlation {corr:.2f}, "
                "and no thermal flag was ever set")
    if pw and pw > 0.5:
        return f"power capped, clock stable ({dec:.1f}% decline)"
    return f"neither limit dominant ({dec:.1f}% clock decline)"


STATES = ("cold", "warm", "recooled")


def rep_state(rep):
    if rep.get("state"):
        return rep["state"]
    if rep.get("cooled"):
        return "recooled"
    return "cold" if rep.get("cold") else "warm"


def worst(xs):
    if not xs:
        return "n/a"
    if any(x is None for x in xs):
        return "never"
    return fmt(max(xs))


def present_states(rows):
    return [s for s in STATES if any(r["state"] == s for r in rows)]


def length_table(rows, lines):
    st = present_states(rows)
    head = "| SKU | repeats | sustained TFLOPS |"
    for s in st:
        head += f" {s}: median s | {s}: worst s |"
    head += " never converged | 60s bias % | median $/hr |"
    lines += [head, "|" + "---|" * (6 + 2 * len(st))]
    for sku in sorted({r["sku"] for r in rows}):
        rs = [r for r in rows if r["sku"] == sku]
        cells = ""
        for s in st:
            xs = [r["t_tol"] for r in rs if r["state"] == s]
            cells += f" {fmt(med(xs))} | {worst(xs)} |"
        lines.append(f"| {sku} | {len(rs)} | {fmt(med([r['sustained'] for r in rs]))} |"
                     f"{cells} {sum(1 for r in rs if r['t_tol'] is None)} of {len(rs)} | "
                     f"{fmt(med([r['bias60'] for r in rs]))} | "
                     f"{fmt(med([r['price'] for r in rs]), 2)} |")


def fleet_table(rows, fleet, ovh, lines):
    lines += [f"| SKU | cold s to tol | overhead s | GPU-hours, test only | $ test only | "
              f"GPU-hours, all in | $ all in | share of the bill that is test time |",
              "|" + "---|" * 8]
    for sku in sorted({r["sku"] for r in rows}):
        rs = [r for r in rows if r["sku"] == sku]
        cold = [r["t_tol"] for r in rs if r["state"] == "cold"]
        t, price, o = med(cold), med([r["price"] for r in rs]), ovh.get(sku)
        gh = fleet * t / 3600 if t else None
        gha = fleet * (t + o) / 3600 if t and o is not None else None
        share = f"{t / (t + o) * 100:.0f}%" if t and o else "unknown"
        lines.append(f"| {sku} | {fmt(t)} | {fmt(o, 0) if o is not None else 'unknown'} | "
                     f"{fmt(gh)} | {fmt(gh * price if gh and price else None, 0)} | "
                     f"{fmt(gha)} | {fmt(gha * price if gha and price else None, 0)} | "
                     f"{share} |")


def overhead_section(runs, lines):
    """Everything billed before the first measured second, which usually dominates."""
    rows = []
    for run in runs:
        t = run["res"].get("timings") or {}
        inf = run["res"].get("inference")
        ml = inf[0].get("model_load_s") if isinstance(inf, list) and inf else None
        a, pstart, fw = t.get("instance_start"), t.get("process_start"), t.get("first_window")
        rows.append({"sku": run["fp"]["gpu"]["name"], "provider": run["fp"]["provider"],
                     "run": run["dir"].name, "price": run["fp"].get("price_per_hr"),
                     "provision": pstart - a if a and pstart else None,
                     "to_first": fw - pstart if fw and pstart else None,
                     "model_load": ml, "total": overhead(t)})
    known = [r for r in rows if r["total"] is not None]
    if not known:
        lines += ["No run recorded an instance start, so overhead is unmeasured and every",
                  "cost figure below counts test time only. Run `setup.sh` first, or pass",
                  "`--instance-start`, or these numbers understate the real bill.", ""]
        return {}
    lines += ["| SKU | provider | run | provision to process s | process to first window s | "
              "model load s | total overhead s |", "|" + "---|" * 7]
    for r in rows:
        lines.append(f"| {r['sku']} | {r['provider']} | {r['run']} | {fmt(r['provision'], 0)} | "
                     f"{fmt(r['to_first'], 0)} | {fmt(r['model_load'], 0)} | "
                     f"{fmt(r['total'], 0)} |")
    by_sku = {}
    for sku in sorted({r["sku"] for r in known}):
        by_sku[sku] = med([r["total"] for r in known if r["sku"] == sku])
    lines += ["", f"- {len(known)} of {len(rows)} runs recorded an instance start.",
              "- Overhead is billed GPU time you cannot benchmark your way out of. Compare it "
              "to the test lengths above before concluding that test length is the cost driver.",
              ""]
    return by_sku


def predictive_power(inst, lines):
    rows = [i for i in inst if i.get("sus") and i.get("tok")]
    if not rows:
        lines.append("No instance has both a compute score and an inference result yet.")
        return
    for i in rows:
        i["ratio"] = i["tok"] / i["sus"]
    lines += ["Delivered output tokens per sustained TFLOPS. If that ratio holds across "
              "instances of a SKU, the compute score can stand in for the workload. If it "
              "moves, the score cannot carry a price.", "",
              "| SKU | instances | median tok/s per TFLOPS | spread across instances | "
              "ranking pairs that agreed |", "|" + "---|" * 5]
    cc = nn = 0
    for sku in sorted({i["sku"] for i in rows}):
        xs = [i for i in rows if i["sku"] == sku]
        rs = [i["ratio"] for i in xs]
        spread = (max(rs) / min(rs) - 1) * 100 if len(rs) > 1 and min(rs) else None
        c = n = 0
        for a in range(len(xs)):
            for b in range(a + 1, len(xs)):
                n += 1
                c += (xs[a]["sus"] - xs[b]["sus"]) * (xs[a]["tok"] - xs[b]["tok"]) > 0
        cc, nn = cc + c, nn + n
        lines.append(f"| {sku} | {len(xs)} | {fmt(med(rs), 2)} | "
                     f"{fmt(spread) + '%' if spread is not None else 'n/a'} | {c} of {n} |")
    errs = []
    for sku in sorted({i["sku"] for i in rows}):
        xs = [i for i in rows if i["sku"] == sku]
        r0 = med([i["ratio"] for i in xs])
        errs += [abs(r0 * i["sus"] - i["tok"]) / i["tok"] * 100 for i in xs]
    lines += ["", f"- Pricing off the compute score: predicting delivered tokens from the SKU "
              f"median ratio is off by {fmt(med(errs))}% at the median, {fmt(max(errs))}% at "
              "the worst. That error is the floor on how precisely a compute certificate can "
              "price inference.",
              f"- Ranking agreement across all within-SKU pairs: {cc} of {nn}."]
    if nn < 6:
        lines.append(f"- {nn} pairs cannot support a correlation coefficient. Two instances per "
                     "SKU gives one pair each. Spearman returns nothing below 3 instances and "
                     "nothing distinguishable from chance below about 5, so none is reported. "
                     "The spread and the pricing error above are what this sample can carry.")
    for sku in sorted({i["sku"] for i in rows}):
        xs = [i for i in rows if i["sku"] == sku]
        if len(xs) >= 5:
            lines.append(f"- {sku}: Spearman rho "
                         f"{fmt(spearman([i['sus'] for i in xs], [i['tok'] for i in xs]), 2)} "
                         f"over {len(xs)} instances.")


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
    stops, curve_rows = [], []

    for run in runs:
        fp, res = run["fp"], run["res"]
        sku = fp["gpu"]["name"]
        for rep in res.get("repeats", []):
            g = rep.get("gemm") or {}
            w, s = g.get("window_tflops"), g.get("window_seconds")
            if not w:
                continue
            # Prefer the raw telemetry CSV over the summary stored at run time: the CSVs
            # are the source of truth, so any later analysis improvement applies
            # retroactively to runs already collected.
            state = rep_state(rep)
            tpath = run["dir"] / f"telemetry_gemm_r{rep['repeat']}.csv"
            tel = summarize_telemetry(tpath) if tpath.exists() else g.get("telemetry", {})
            common = {"sku": sku, "provider": fp["provider"], "run": run["dir"].name,
                      "repeat": rep["repeat"], "state": state, "price": fp.get("price_per_hr")}
            for name, warm in estimators:
                c = convergence(w, s, tols=tols, warmup=warm)
                if not c:
                    continue
                for t, e in zip(c["t_s"], c["errors_pct"]):
                    curve_rows.append([name, sku, fp["provider"], run["dir"].name,
                                       rep["repeat"], state, round(t, 3), round(e, 4)])
                per_run[name].append(dict(
                    common, estimator=name, uuid=fp["gpu"].get("uuid"),
                    cold=state == "cold", power_limit=fp["gpu"].get("power.limit"),
                    driver=fp["gpu"].get("driver_version"),
                    sustained=c["reference_tflops"], run_length_s=c["total_s"],
                    t1=c["t_converge_1pct_s"], t2=c["t_converge_2pct_s"],
                    t5=c["t_converge_5pct_s"], t_tol=c[key],
                    bias60=c["short_test_bias_pct_at_60s"],
                    power_cap_frac=tel.get("throttle_fraction", {}).get("power_cap"),
                    thermal_frac=tel.get("throttle_fraction", {}).get("thermal"),
                    membw=(rep.get("membw") or {}).get("median_gbps")))
            sr = sequential_stop(w, s, tol=args.tol, window_s=args.stop_window,
                                 batches=args.stop_batches, confirm=args.stop_confirm)
            sc = settled_check(w, s, tel)
            if sr:
                stops.append(dict(common, settled=sc, **sr))

    rev = sh("git -C " + str(Path(__file__).resolve().parent) + " rev-parse --short HEAD") or "unversioned"
    dirty = " +uncommitted" if sh("git -C " + str(Path(__file__).resolve().parent)
                                  + " status --porcelain -- gpubench.py") else ""
    lines = ["# gpubench summary", "",
             f"Produced by gpubench `{rev}{dirty}` on "
             f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}. "
             f"Settings: tol {args.tol:g}%, warmup {args.warmup:g}s, stop-window "
             f"{args.stop_window:g}s, stop-batches {args.stop_batches}, stop-confirm "
             f"{args.stop_confirm}, fleet {args.fleet}. Numbers are reproducible only "
             "against this revision and these settings: the analysis re-derives telemetry "
             "from the raw CSVs, so a later revision can legitimately change a published "
             "figure.", "",
             f"Tolerance {args.tol:g}%. Benchmark length is total wall time on the GPU, "
             f"warm-up included. Fleet figures assume {args.fleet:,} GPUs.", "",
             "Three ways of turning a run into one number are compared. The first two are "
             "**oracles**: they use the full run to say what the shortest sufficient test "
             "would have been, which nobody can know while the test is running. The third is "
             "**causal** and is the only one that could be deployed.", ""]

    lines += [f"## 1. Oracle: length needed to land within {args.tol:g}% of steady state", ""]
    for name, warm in estimators:
        lines += [f"### `{name}`: " + ("average from second one" if not warm
                                       else f"discard the first {warm:g} s"), ""]
        length_table(per_run[name], lines)
        lines.append("")
    lines += [f"A warm-up estimator cannot report a test shorter than its own warm-up, so "
              f"`{names[1]}` floors at about {args.warmup:g} s. Where a SKU shows that floor, "
              "the estimator is setting the length, not the card.", "",
              "Positive 60s bias means a one-minute test overstates steady-state throughput. "
              "It is blank for runs shorter than 60 s.", ""]

    lines += ["## 2. Causal: a sequential stopping rule you could ship", "",
              f"Stops when the 95% confidence interval on the last {args.stop_window:g} s "
              f"(batch means, {args.stop_batches} batches) is under {args.tol:g}% and "
              f"the trend across it is not distinguishable from zero at 95%, for "
              f"{args.stop_confirm} consecutive checks. No extrapolation: you cannot certify "
              "stability over a horizon longer than you observed, so durability is a "
              "repeat-measurement question, not a longer-run one.", ""]
    if stops:
        lines += ["| SKU | state | runs | stopped | median stop s | worst stop s | "
                  "median realized error % | worst realized error % | stopped early (over tol) |",
                  "|" + "---|" * 9]
        for sku in sorted({r["sku"] for r in stops}):
            for st in present_states(stops):
                xs = [r for r in stops if r["sku"] == sku and r["state"] == st]
                if not xs:
                    continue
                done = [r for r in xs if r["stopped"]]
                es = [r["realized_error_pct"] for r in done]
                lines.append(
                    f"| {sku} | {st} | {len(xs)} | {len(done)} | "
                    f"{fmt(med([r['t_stop_s'] for r in done]))} | "
                    f"{fmt(max([r['t_stop_s'] for r in done]) if done else None)} | "
                    f"{fmt(med(es), 2)} | {fmt(max(es) if es else None, 2)} | "
                    f"{sum(1 for r in done if not r['within_tol'])} |")
        lines += ["", "| SKU | run | rep | second-half trend %/min | trend t | SM MHz mean/min | "
                  "power W | temp C | power-cap frac | thermal frac | clock decline % | "
                  "clock/temp corr | which limit | verdict |",
                  "|" + "---|" * 14]
        for r in stops:
            sc = r.get("settled") or {}
            lines.append(f"| {r['sku']} | {r['run']} | {r['repeat']} | "
                         f"{fmt(sc.get('second_half_trend_pct_per_min'), 4)} | "
                         f"{fmt(sc.get('trend_t'), 2)} | {fmt(sc.get('sm_mhz_mean'), 0)}/"
                         f"{fmt(sc.get('sm_mhz_min'), 0)} | {fmt(sc.get('power_w_mean'), 0)} | "
                         f"{fmt(sc.get('temp_c_max'), 0)} | {fmt(sc.get('power_cap_frac'), 2)} | "
                         f"{fmt(sc.get('thermal_frac'), 2)} | {fmt(sc.get('sm_decline_pct'), 1)} | "
                         f"{fmt(sc.get('clock_temp_corr'), 2)} | {sc.get('limit', 'n/a')} | "
                         f"{sc.get('verdict', 'n/a')} |")
        lines += ["", "The stopping rule is a candidate, not a verdict. It judges only the last "
                  f"{args.stop_window:g} s, so it has no power against drift slower than roughly "
                  "the per-window noise divided by that span: on the control case of a card "
                  "sliding 8% across a run, it stops at 61 s and is 7.3% wrong. The whole-run "
                  "check above uses every sample and catches that case at t=104. A rating is "
                  "only defensible when both agree.", ""]
        lines += ["", f"The rule cannot answer before {args.stop_window:g} s, since that is the "
                  "span the precision test reads. Where an oracle converges faster than that, the "
                  "floor is the statistics rather than the silicon, and it is a cheap floor. "
                  "'Stopped early' counts runs where the rule fired but the answer was outside "
                  "tolerance; those are false confidence and matter more than a slow stop.", ""]
    else:
        lines += ["No run was long enough to evaluate the stopping rule.", ""]

    lines += ["## 3. Which rule to use", "",
              "| SKU | state | " + " | ".join(f"{n} (oracle)" for n in names)
              + " | sequential stop (causal) | realized error at stop % |",
              "|" + "---|" * (4 + len(names))]
    for sku in sorted({r["sku"] for r in per_run[names[0]]}):
        for st in present_states(per_run[names[0]]):
            sel = {n: [r["t_tol"] for r in per_run[n]
                       if r["sku"] == sku and r["state"] == st] for n in names}
            if not any(sel.values()):
                continue
            sx = [r for r in stops if r["sku"] == sku and r["state"] == st and r["stopped"]]
            lines.append(f"| {sku} | {st} | " + " | ".join(fmt(med(sel[n])) for n in names)
                         + f" | {fmt(med([r['t_stop_s'] for r in sx]))} | "
                         f"{fmt(med([r['realized_error_pct'] for r in sx]), 2)} |")
    lines.append("")

    # Thermal or software: only answerable when a cooldown repeat exists
    recooled = [r for r in per_run[names[0]] if r["state"] == "recooled"]
    lines += ["## 4. Is the cold penalty thermal or software?", ""]
    if not recooled:
        lines += ["No run used --cooldown, so repeat 0 confounds a cold card with a cold "
                  "process (cuBLAS autotune, first-touch allocation). Until one does, a cold "
                  "penalty cannot be attributed to temperature.", ""]
    else:
        lines += ["| SKU | cold s | warm s | cold-again s | reading |", "|" + "---|" * 5]
        for sku in sorted({r["sku"] for r in recooled}):
            g = {s: med([r["t_tol"] for r in per_run[names[0]]
                         if r["sku"] == sku and r["state"] == s]) for s in STATES}
            verdict = "inconclusive"
            if all(g[s] is not None for s in STATES) and g["cold"] != g["warm"]:
                frac = (g["recooled"] - g["warm"]) / (g["cold"] - g["warm"])
                verdict = ("thermal: the penalty comes back when the card cools" if frac > 0.6
                           else "software: a one-time per-process cost, not temperature"
                           if frac < 0.3 else "mixed")
            lines.append(f"| {sku} | {fmt(g['cold'])} | {fmt(g['warm'])} | "
                         f"{fmt(g['recooled'])} | {verdict} |")
        for run in runs:
            cd = run["res"].get("cooldown")
            if cd:
                t = cd.get("telemetry", {})
                lines.append(f"- {run['dir'].name}: idled {cd['minutes']:g} min, temperature "
                             f"{fmt(t.get('temp_c_first'), 0)} C to {fmt(t.get('temp_c_last'), 0)} C")
        lines.append("")

    lines += ["## 5. Where the billed time actually goes", ""]
    ovh = overhead_section(runs, lines)

    lines += [f"## 6. Cost to certify {args.fleet:,} GPUs at {args.tol:g}%", ""]
    if stops:
        lines += ["### Causal: sequential stop. This is the one to quote.", ""]
        fleet_table([dict(r, t_tol=r["t_stop_s"], sustained=r["reference_tflops"])
                     for r in stops], args.fleet, ovh, lines)
        lines.append("")
    for name, warm in estimators:
        lines += [f"### Oracle `{name}`: a lower bound, not a plan", ""]
        fleet_table(per_run[name], args.fleet, ovh, lines)
        lines.append("")
    lines += ["Cold time-to-converge is also the idle time each seller gives up per card. "
              "The cold column is a median over the cold runs that converged; runs that never "
              "converged are counted in the length tables, not averaged into this number.", ""]

    lines += ["## 7. Per run", "",
              "| SKU | provider | run | rep | state | power W | driver | sustained | run len s | "
              + " | ".join(f"t {n}" for n in names) + " | t stop | t1% | t5% | power-cap frac | "
              "thermal frac | mem GB/s |", "|" + "---|" * (15 + len(names))]
    other = {(r["run"], r["repeat"]): r for r in per_run[names[1]]}
    stopmap = {(r["run"], r["repeat"]): r for r in stops}
    for r in per_run[names[0]]:
        k = (r["run"], r["repeat"])
        lines.append(f"| {r['sku']} | {r['provider']} | {r['run']} | {r['repeat']} | "
                     f"{r['state']} | {r['power_limit']} | {r['driver']} | "
                     f"{fmt(r['sustained'])} | {fmt(r['run_length_s'], 0)} | {fmt(r['t_tol'])} | "
                     f"{fmt(other.get(k, {}).get('t_tol'))} | "
                     f"{fmt(stopmap.get(k, {}).get('t_stop_s'))} | {fmt(r['t1'])} | "
                     f"{fmt(r['t5'])} | {fmt(r['power_cap_frac'], 2)} | "
                     f"{fmt(r['thermal_frac'], 2)} | {fmt(r['membw'], 0)} |")

    lines += ["", "## 8. Does the compute score predict inference throughput?", ""]
    inst = []
    for run in runs:
        inf, reps = run["res"].get("inference"), run["res"].get("repeats", [])
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
    predictive_power(inst, lines)
    if inst:
        lines += ["", "| SKU | provider | run | sustained TFLOPS | out tok/s | $/hr | $/M out tok |",
                  "|---|---|---|---|---|---|---|"]
        for i in inst:
            lines.append(f"| {i['sku']} | {i['provider']} | {i['run']} | {fmt(i['sus'])} | "
                         f"{fmt(i['tok'], 0)} | {fmt(i['price'], 2)} | {fmt(i['usd_m'], 3)} |")

    by_uuid = {}
    for run in runs:
        inf = run["res"].get("inference")
        if isinstance(inf, list) and inf:
            by_uuid.setdefault(run["fp"]["gpu"].get("uuid"), []).append(
                (run["fp"].get("label", "default"), run["fp"]["software"].get("vllm"),
                 run["fp"]["software"].get("torch"),
                 med([i["output_tokens_per_s"] for i in inf])))
    sw = {u: v for u, v in by_uuid.items() if len({x[0] for x in v}) > 1}
    lines += ["", "## 9. Same GPU, different software stack", ""]
    if not sw:
        lines.append("No GPU was run under more than one --label yet.")
    for u, v in sw.items():
        base = v[0][3]
        lines.append(f"- GPU {u[-8:] if u else '?'}:")
        for label, vv, tv, tok in v:
            lines.append(f"  - {label} (vllm {vv}, torch {tv}): {fmt(tok, 0)} tok/s, "
                         f"{fmt((tok - base) / base * 100 if base and tok else None)}% vs first")

    lines += ["", "## 10. Data quality flags", ""]
    flags = []
    for r in per_run[names[0]]:
        tag = f"{r['sku']} {r['run']} rep {r['repeat']}"
        if r["run_length_s"] and r["run_length_s"] < 300:
            flags.append(f"- {tag}: run only {r['run_length_s']:.0f} s, too short to establish "
                         "steady state")
        if r["t_tol"] is None:
            flags.append(f"- {tag}: never converged, so its steady state is an assumption")
        if r["bias60"] is None:
            flags.append(f"- {tag}: shorter than 60 s, no 60s bias available")
    for r in stops:
        sc = r.get("settled") or {}
        if r["stopped"] and sc.get("verdict", "").startswith("still moving"):
            flags.append(f"- {r['sku']} {r['run']} rep {r['repeat']}: the stopping rule fired but "
                         f"the whole run is still moving ({fmt(sc.get('second_half_trend_pct_per_min'), 4)}"
                         "%/min, t=" + fmt(sc.get("trend_t"), 1) + "). Do not rate this run.")
        if r["stopped"] and not r["within_tol"]:
            flags.append(f"- {r['sku']} {r['run']} rep {r['repeat']}: the stopping rule fired at "
                         f"{r['t_stop_s']:.0f} s but was {r['realized_error_pct']:.1f}% off. "
                         "False confidence: widen --stop-window or raise --stop-batches.")
        if not r["stopped"]:
            flags.append(f"- {r['sku']} {r['run']} rep {r['repeat']}: the stopping rule never "
                         "fired within the run")
    for run in runs:
        if not (run["res"].get("timings") or {}).get("instance_start"):
            flags.append(f"- {run['dir'].name}: no instance start recorded, overhead unknown")
        inf = run["res"].get("inference")
        if isinstance(inf, dict) and inf.get("skipped"):
            flags.append(f"- {run['dir'].name}: inference skipped ({inf['skipped']})")
        for rep in run["res"].get("repeats", []):
            if not ((rep.get("gemm") or {}).get("telemetry") or {}).get("samples", 0):
                flags.append(f"- {run['dir'].name} rep {rep.get('repeat')}: GEMM telemetry "
                             "empty, throttle attribution unavailable")
    lines += sorted(set(flags)) or ["None."]

    text = "\n".join(lines)
    Path(args.dir, "summary.md").write_text(text)
    Path(args.dir, "summary.json").write_text(json.dumps(
        {"tol": args.tol, "warmup": args.warmup, "fleet": args.fleet, "estimators": names,
         "stop_rule": {"window_s": args.stop_window, "batches": args.stop_batches,
                       "confirm": args.stop_confirm},
         "runs": per_run, "stops": stops, "overhead_by_sku": ovh, "instances": inst}, indent=2))
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
    r.add_argument("--cooldown", type=float, default=0.0,
                   help="idle this many minutes after the repeats, then run one more "
                        "(cold-again): separates a thermal cold penalty from a software one")
    r.add_argument("--fingerprint-only", action="store_true",
                   help="record the configuration fingerprint and exit, for surveying what a "
                        "host actually provides without paying for a benchmark")
    r.add_argument("--instance-start", default=None,
                   help="unix time or ISO timestamp at which you launched the instance, "
                        "so the summary can price provisioning overhead, not just test time")
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
    a.add_argument("--stop-window", type=float, default=60.0,
                   help="seconds of recent data the sequential rule judges. Shorter windows "
                        "cannot estimate the trend precisely enough to project it")
    a.add_argument("--stop-batches", type=int, default=6,
                   help="batches used for the confidence interval, to handle autocorrelation")
    a.add_argument("--stop-confirm", type=int, default=2,
                   help="consecutive passing checks before the rule stops")
    a.set_defaults(func=cmd_analyze)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
