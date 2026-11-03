# The scoring rule is the benchmark

How long a GPU benchmark has to run, and why that question cannot be answered until you fix
how the score is computed. [N] instances of H100 SXM, A100 and L40S across [providers].
Sean Mihm · [date] · code and raw data: https://github.com/maker-sean/gpubench

## Bottom line
- **Fix the scoring rule before the test length.** Averaging a run from second one bakes the
  warm-up into the score, so the required length falls out as roughly 50 x drop x settling
  time: a number you can derive from your own assumptions without renting a GPU. [Rule] stops
  instead when the confidence interval on recent throughput is tight and the trend is flat. It
  uses only past data, so unlike the usual answer it can actually run live, and it ships an
  interval with every rating rather than a bare number.
- **Under that rule, required length is a SKU property and it varies [N]x.** Cold [SKU A]
  stopped at [X] s, cold [SKU B] at [Y] s ([Z] s on the worst instance). Warm cards stopped at
  [W] s. A fixed 60 s test overpays [A] s per card on [fast SKU] and under-tests [slow SKU] by
  [B] s. Note the floor: an oracle with hindsight says [SKU A] only needed [X_oracle] s, but
  proving convergence takes [stop window] s, so that is the shortest defensible test.
- **A short test sells a number the buyer realizes only at low duty cycle.** A 60 s test on a
  cold [SKU] read [B]% above steady state. A buyer running back-to-back jobs gets the
  steady-state number; the certificate priced the cold one. At [$]/hr that gap is [$] per
  card-year of compute paid for and not received, which is a pricing error, not a rounding one.
- **Test length is not where the money is.** Provisioning to first measured second took [O] s
  at the median, [P]x the measurement itself. Test time is [S]% of the cost of certifying
  10,000 [SKU]. Optimizing the test without optimizing the pipeline moves [small]% of the bill.

[CHART: error vs benchmark length, one line per SKU, cold solid and warm dashed, shaded
tolerance band, stop times marked]

| SKU | Sustained TFLOPS | Cold stop time | Realized error | 60 s bias | Test share of cost |
|---|---|---|---|---|---|
| H100 SXM | | | | | |
| A100 | | | | | |
| L40S | | | | | |

## Method
[2 to 3 sentences: instances, providers, repeats, 20 minute bf16 GEMM with 1 s windows,
telemetry and configuration fingerprint per run, cooldown repeat to separate thermal from
software warm-up.] [1 sentence: three scoring rules compared, two oracle and one causal, and
why the causal one is the only candidate.] Limits: single GPU, [N] instances, no multi-node,
hours not months.

## What this means for SiliconMark
- **Certify on a stopping rule, not a clock.** Same principle as sequential testing: stop when
  the interval is tight enough. Cheaper on [fast SKU], more trustworthy on [slow SKU].
- **Publish the interval with the rating.** A certificate that says [X] TFLOPS +/- [Y]% is
  tradeable. One that says [X] is an assertion.
- **Reject unsettled runs.** [N] of [M] runs never reached steady state. Flag them from clock
  and throttle telemetry so a short cold test cannot be passed off as a rating.
- **Bind certificates to the software stack.** On the same physical H100, changing only
  [vLLM x to y] moved inference throughput [D]%. Name the stack on the certificate.
- **Price the pipeline, not the test.** Report cost and time to first valid rating end to end.

## Next
Drift over weeks on the same devices (I ran an accelerated decay study for DNA storage at
Catalog; certificates need the same treatment), and multi-node cohorts.
