# How short can a defensible GPU benchmark be?

Test length, confidence, and SKU, measured across [N] instances of H100, A100, and L40S on [providers].
Sean Mihm · [date] · code and raw data: [repo link]

## Bottom line
- **Test length depends on SKU.** To land within 2% of steady state, a cold [SKU A] needed [X] s and a cold
  [SKU B] needed [Y] s ([Z] on the worst instance). Warm cards needed [W] s or less.
- **Short tests overstate.** A 60 second test on a cold [SKU] read [B]% above steady state. A seller who
  benchmarks a fresh card for one minute certifies performance a buyer will not get.
- **Certificates have a cost.** Certifying 10,000 [SKU] cards to 2% confidence costs about [GPU-hours] of
  idle time, or [$]. A fixed-length test overpays on fast-settling SKUs and misrates slow ones.

[CHART: error vs benchmark length, one line per SKU, cold solid and warm dashed, shaded 2% band]

| SKU | Sustained TFLOPS | Cold s to 2% | Warm s to 2% | 60 s bias | Cost to certify 10k |
|---|---|---|---|---|---|
| H100 SXM | | | | | |
| A100 | | | | | |
| L40S | | | | | |

## Method
[2 to 3 sentences: instances, providers, repeats, 20 minute bf16 GEMM with 1 s windows, telemetry and
configuration fingerprint per run.] [1 sentence on the estimator decision: average from start vs discard
warm-up, and which one holds up.] Limits: single GPU, [N] instances, no multi-node, hours not months.

## What this means for SiliconMark
- **Adaptive test length.** Stop each run when the confidence interval is tight enough, not at a fixed
  time. Same principle as sequential A/B testing: cheaper on [fast SKU], more trustworthy on [slow SKU].
- **Reject unsettled runs.** Flag results from cards that never reached steady state, using clock and
  throttle telemetry, so a short cold test cannot be passed off as a rating.
- **Bind certificates to software.** On the same H100, changing only [vLLM x to y] moved inference
  throughput [Δ]%. A certificate should name the stack it was measured on.
- **Track cost per certificate** alongside time to first result.

## Next
Drift over weeks on the same devices (I ran an accelerated decay study for DNA storage at Catalog;
certificates need the same treatment), and multi-node cohorts.
