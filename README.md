# gpubench

**How short can a defensible GPU benchmark be, and does the answer depend on the SKU?**

A benchmark that runs for 60 seconds on a freshly allocated cloud GPU does not measure the
same thing as one that runs for 20 minutes. This repo measures the gap: it records 1 second
throughput windows for a long sustained run, then asks what a test of any shorter length
would have reported and how far off it would have been.

Every run also captures a configuration fingerprint and 500 ms telemetry, so a difference
between two machines of the same SKU can be attributed rather than just observed.

> **Status: harness complete, measurement runs pending.** `results/` is empty until real
> instances are benchmarked. No numbers in this repo are modeled, extrapolated or estimated;
> anything published here will be reproducible from the raw data committed alongside it.

## What it measures

| Test | Metric | Why it matters |
|---|---|---|
| bf16 GEMM held 20 min, 1 s windows | time to converge, 60 s bias, sustained median, decay %, CV | How long a test must run before its answer stops moving |
| Device memory copy | GB/s (read + write) | HBM bandwidth, the bottleneck for most inference |
| vLLM, fixed prompts, fixed output length | output tokens/s, $/M tokens | Delivered value in economic terms |

Fingerprint per run: GPU name, UUID, serial, VBIOS, driver, CUDA, power limit (current,
default, max), max clocks, ECC, MIG, PCIe gen and width, topology, host CPU.
Telemetry every 500 ms: SM and memory clocks, power draw, temperature, throttle reasons.

## Definitions

- **Steady state (reference)**: median throughput over the final 25% of a run.
- **Estimate at t**: time weighted mean throughput over a test of length t, that is, what a
  t second benchmark would have reported.
- **Time to converge at tol**: earliest t after which the estimate stays within tol% of the
  reference. Default tol is 2%. "never" means it did not converge within the run.
- **60 s bias**: (estimate at 60 s minus reference) / reference. Positive means a short test
  overstates.

Benchmark length `t` is always **total wall clock time on the GPU, warm-up included**. A test
that discards its first 30 seconds still occupies the card for them, and still pays for them.

### Two estimators, reported side by side

Averaging from second one bakes the warm-up into the number, so a slow settling card can take
a long time to converge for reasons that are arithmetic rather than physical. For a card
decaying with drop `d` and time constant `tau`, that estimator converges at roughly
`t = 100 * d * tau / tol`, which scales with the tolerance you picked and the shape of the
ramp, not with anything a buyer experiences.

`analyze` therefore reports both estimators:

- `from_start`: average from second one. What a naive test reports.
- `warmup<K>s`: discard the first K seconds (`--warmup`, default 30), then average.

Neither is obviously right. The warm-up estimator cannot report a test shorter than its own
warm-up, so it floors at K seconds and is *worse* than `from_start` on fast settling SKUs.
Reporting both is how you tell a property of the card from a property of the arithmetic.

## Setup (on each instance)

```bash
git clone https://github.com/maker-sean/gpubench && cd gpubench
bash setup.sh --prefetch
```

`setup.sh` installs vLLM (which pins a compatible torch), prints the GPU name, driver, power
limit against its default limit, PCIe link and `CUDA_VISIBLE_DEVICES`, optionally
pre-downloads the model so the download is not billed inside the timed run, then runs a short
smoke test into a separate `results-smoke/` directory.

## Run

```bash
python gpubench.py run --provider runpod --price-per-hr 2.69
```

Software stack comparison on a box that has already been benchmarked:

```bash
python gpubench.py run --provider runpod --inference-only --label vllm-0.9.2
```

Copy each `results/<provider>/<run>/` directory back to one machine, then:

```bash
python gpubench.py analyze --dir results --tol 2 --warmup 30
python plot_convergence.py --estimator both
```

`analyze` writes `summary.md`, `summary.json` and `convergence.csv`.
`plot_convergence.py` writes `convergence.png`: error against benchmark length, one color and
marker per SKU, cold solid and warm dashed, a light band for the range across instances, and a
shaded band at the tolerance.

## Reading the results

- **Time to converge**: the headline. Compared by SKU, cold against warm, and under both
  estimators. Quote it with the estimator named; the number is meaningless without it.
- **60 s bias**: how far a one minute test lands from steady state. Positive means it
  overstates sustained throughput.
- **Cost to certify**: GPU-hours and dollars to take a fleet to the tolerance, against what a
  fixed 60 second test would cost, and whether that fixed test actually converges.
- **decay %**: first 5 seconds against the sustained second half. High decay with a high
  power-cap or thermal fraction means the box cannot hold clocks.
- **Data quality flags**: runs that never converged, runs too short to establish steady state,
  empty telemetry, skipped inference. Read this section before quoting any number above it.
- **Fingerprint diffs**: when two runs of the same SKU differ, compare power limit, driver,
  PCIe link and form factor first. A provider that has lowered `power.limit` below
  `power.default_limit` has capped the card.

## Limitations

- Single GPU only. No NVLink or InfiniBand, no multi-node collectives.
- Small sample: a handful of devices per provider. Rank correlation between the synthetic
  score and inference throughput needs at least 3 instances of a SKU to produce a number.
- One model and one prompt shape for inference.
- Hours, not months. This measures what changes a measurement, not how a rating drifts over
  time. Long term drift is a separate experiment.
- The cost figures count GPU time spent measuring. They do not include provisioning, image
  pull, model download or queue wait, which in practice dominate the cost of certifying a
  machine.

## License

MIT. See [LICENSE](LICENSE).
