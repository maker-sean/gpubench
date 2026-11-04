# How long a benchmark has to run, measured across four cards

Six runs, four SKUs, two instances each of the H100 SXM and the L40S. 27,448 convergence
points. "Required length" is the last moment the running estimate is still outside 2% of
the run's own steady state — that is, the shortest test you could have run and been inside
tolerance.

Charts: `convergence_from_start.png`, `convergence_warmup30s.png`, `convergence_both.png`.

## Required test length to land within 2% of steady state

Estimator `from_start` — average every window from the second the benchmark starts. This is
what a test that just runs and reports does.

| SKU | state | n | median | min | max |
|---|---|---|---|---|---|
| H100 80GB HBM3 (SXM) | cold | 2 | **0.0 s** | 0.0 | 0.0 |
| H100 PCIe | cold | 1 | 3.1 s | - | - |
| H100 NVL | cold | 1 | 4.0 s | - | - |
| L40S | cold | 2 | **264.8 s** | 78.2 | 451.5 |
| L40S | warm | 4 | 45.3 s | 0.0 | 433.9 |
| L40S | recooled | 2 | 674.6 s | 148.6 | 1200.6* |

\* right-censored: the run itself ended at 1200 s, so that instance never came inside
tolerance within the measurement.

The H100 SXM is inside 2% from its first measured second — it never leaves tolerance, so
its requirement is below the 1 s resolution of the measurement, not zero.

**The honest spread: about 3 seconds to about 4.5 minutes, a factor of 85, across cards a
buyer finds in one provider's catalogue.** Reporting it as a ratio against the SXM would
give a meaningless number, because the SXM's requirement is below the measurement floor.

## What discarding warm-up fixes, and what it does not

Estimator `warmup30s` — throw away the first 30 seconds, then average.

| SKU | state | median required |
|---|---|---|
| H100 80GB HBM3 | cold | 0.0 s |
| H100 PCIe | cold | **0.0 s** (from 3.1) |
| H100 NVL | cold | **0.0 s** (from 4.0) |
| L40S | cold | **185.7 s** (from 264.8) |
| L40S | warm | 24.6 s (from 45.3) |
| L40S | recooled | 601.5 s (from 674.6) |

Discarding 30 s of warm-up completely solves all three H100s and leaves the L40S needing
three more minutes. The reason is the ramp length: the H100s reach their power-capped
equilibrium within seconds, so 30 s is more than enough, while the L40S's thermal ramp is
longer than the window being discarded. A fixed warm-up constant is a SKU-specific
assumption dressed as a general one.

## Same card, different thermal state

The L40S is the only SKU where the three states separate, and they separate by a lot:

- cold 264.8 s
- warm 45.3 s
- recooled 674.6 s

Cold needs about 6x the warm requirement. The recooled figure is worse than cold and is
dominated by one unstable instance (L40S #1); the other recooled repeat came inside
tolerance at 149 s. This is the same host-stability effect reported in the repeatability
result, seen through convergence instead of through spread.

## What follows for a fixed-length test

A test with one duration for every card is either too long for the H100s or too short for
a cold L40S. At "a few minutes" it is roughly 100x longer than an H100 SXM needs and, on a
cold L40S, lands near or inside the part of the curve where the error is still moving.

The alternative is not a longer fixed test. It is a stopping rule: run until the estimate
is stable, then report, and report how long that took. That number — time to stability —
is itself a property worth certifying, because it is the one that varies 85x across the
catalogue while the marketing name stays the same.

## Caveats

- Two instances each for the H100 SXM and L40S; one each for the NVL and PCIe. The single
  instances carry no spread, so their figures are points, not ranges.
- "Steady state" is each run's own late-run mean, so this measures convergence to that
  run's equilibrium, not agreement between runs. Cross-run agreement is reported separately.
- The L40S recooled maximum is censored at the run length and the true value is larger.
