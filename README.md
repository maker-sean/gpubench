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

### Three scoring rules, reported side by side

Averaging from second one bakes the warm-up into the number, so a slow settling card takes a
long time to converge for reasons that are arithmetic rather than physical. For a card
decaying with drop `d` and time constant `tau`, that rule converges at roughly
`t = 100 * d * tau / tol`: a figure you can derive from your own assumptions without renting
anything, and one that moves by a third if you change the tolerance in a spreadsheet.

`analyze` therefore reports three rules, and the distinction between them matters more than
any of their numbers:

| Rule | What it is | Deployable? |
|---|---|---|
| `from_start` | average from second one | oracle |
| `warmup<K>s` | discard the first K seconds (`--warmup`, default 30), then average | oracle |
| sequential stop | stop when the interval is tight and the trend is flat | **causal** |

The first two are **oracles**. They use the complete run to say what the shortest sufficient
test *would have been*, which nobody can know while a test is running. They are lower bounds,
useful for showing how much room exists, not candidate rules.

The sequential rule is **causal**: at each second it looks only at the past, and stops when
two things hold together for `--stop-confirm` consecutive checks.

- **Precise.** The 95% confidence interval on the mean of the last `--stop-window` seconds is
  inside the tolerance. It is built from batch means rather than raw per-second samples,
  because consecutive seconds on one GPU are correlated and a naive standard error would claim
  confidence the data does not support.
- **Settled.** The trend across that window is not statistically distinguishable from zero.

Neither check asks how long the answer will stay true. An earlier version of this rule
projected the window trend forward over a horizon and refused to rate a card whose answer
might move. That was a mistake worth recording: on a flat card the projection is a noisy slope
multiplied by the horizon, so it inflates with the length of the promise rather than with
anything the hardware did, and it made the stop time swing by a factor of two on the choice of
batch count. **You cannot certify stability over a horizon longer than you observed.**
Durability is a repeat-measurement question, not a longer-run one.

### The stopping rule is a candidate, not a verdict

The rule reads only the last `--stop-window` seconds, so it has no power against drift slower
than roughly the per-window noise divided by that span. On a control case of a card sliding 8%
across its run it stops at 61 s and is 7.3% wrong.

`analyze` therefore also runs a **whole-run settled check** on every repeat, using every
sample: a trend test on the second half of the throughput series, reported next to clock,
power, temperature and throttle fractions from telemetry. That check catches the same control
case at t=104. A rating is defensible only when both agree, and `analyze` flags any run where
the rule fired but the whole run is still moving.

## Setup (on each instance)

```bash
git clone https://github.com/maker-sean/gpubench && cd gpubench
bash setup.sh --prefetch
```

`setup.sh` installs vLLM (which pins a compatible torch), prints the GPU name, driver, power
limit against its default limit, PCIe link and `CUDA_VISIBLE_DEVICES`, optionally
pre-downloads the model so the download is not billed inside the timed run, then runs a short
smoke test into a separate `results-smoke/` directory.

It also writes `.gpubench_instance_start`, a timestamp used to price everything billed before
the first measured second. That stamp is a lower bound, because it starts when you got a
shell, not when the instance started charging. Pass the real value if you have it:

```bash
python gpubench.py run --provider runpod --price-per-hr 2.69 --instance-start 1762070400
```

## Run

```bash
python gpubench.py run --provider runpod --price-per-hr 2.69
```

Separate a thermal cold penalty from a one-time software one, by idling the card and running
once more (adds the idle time to the bill, so use it on one instance per SKU, not all six):

```bash
python gpubench.py run --provider runpod --price-per-hr 2.69 --cooldown 15
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

## Before committing to a 20 minute default

The 20 minute GEMM is a guess. If a card has not settled within it there is no steady state to
compare against and every number for that run reads "never". Spend one hour on the first and
slowest SKU before standardising:

```bash
python gpubench.py run --provider runpod --price-per-hr 2.69 \
  --repeats 1 --gemm-seconds 2700 --skip-inference --out results-explore
python gpubench.py analyze --dir results-explore
```

If the cold run converges well inside 2700 s, 1200 s is safe. If it does not, raise
`--gemm-seconds` for every instance before spending the budget, because a run that never
settles cannot be rescued in analysis.

## Reading the results

- **Stop time and realized error**: the headline pair. A stop time without its realized error
  says nothing about whether the rule works. Quote both, and name the rule.
- **Time to converge**: the oracle lower bound, by SKU and by state. Useful for showing the gap
  between what was sufficient and what is provable. Never quote it as a recommendation.
- **Overhead**: provisioning to first measured second. Compare it to the test lengths before
  concluding that test length drives cost; usually it does not.
- **Cold against cold-again**: with `--cooldown`, whether the cold penalty returns after the
  card is idled. If it does, it is thermal. If it does not, repeat 0 was paying a one-time
  software cost and the "cold" framing is wrong.
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
- Small sample: a handful of devices per provider. Two instances per SKU gives one ranking
  pair per SKU, which cannot support a correlation coefficient, so none is reported. What the
  sample can carry is the spread in delivered tokens per TFLOPS and the error from pricing off
  the compute score, and those are what `analyze` prints.
- One model and one prompt shape for inference.
- Hours, not months. This measures what changes a measurement, not how a rating drifts over
  time. Long term drift is a separate experiment.
- The cost figures count GPU time spent measuring. They do not include provisioning, image
  pull, model download or queue wait, which in practice dominate the cost of certifying a
  machine.

## License

MIT. See [LICENSE](LICENSE).
