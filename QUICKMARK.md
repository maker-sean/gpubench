# Running SiliconMark's QuickMark against its own claims

Measured 2026-11-04 on one NVIDIA L40S (third distinct card of this study), rented from
Runpod Secure, host reporting location AU. Agent: the stable build from
`downloads.silicondata.com/agent`, dated 2026-09-28, run unregistered with no account and
no job key. Three QuickMark runs on the same card over two hours — one cold, one after a
600 s GEMM soak, and one after a ten-minute idle cooldown — with the card sampled at 1 Hz
throughout by an independent process.

## Access

Free. Silicon Data's Basic tier costs nothing and needs no credit card, and the User Guide
documents an unregistered path (`./agent`) that runs the benchmark without an account.
No trial, no sales call.

One correction worth recording, because the docs invite the mistake: `--setup` is a flag
that only builds the agent's private environment and exits rc=0. The unregistered run is
the bare `./agent`. Running `--setup` alone looks like success and benchmarks nothing.

In unregistered mode the only result that reaches the terminal is the SM Score letter. No
per-GPU TFLOPS, bandwidth or temperature figures are written locally; those go to the
portal. That limits what an outside observer can check to the letter plus their own
instrumentation.

## Result: the score did not move

| run | card before the run | speed pass | thermal probe | SM Score |
|---|---|---|---|---|
| cold | 31 C, idle | 403 s | 702 s | **D** (scoring v1, QM v2.0) |
| warm | 62 C, after a 600 s soak | 406 s | 699 s | **D** (scoring v1, QM v2.0) |
| recooled | 30 C, after a 10 min idle | 399 s | 688 s | **D** (scoring v1, QM v2.0) |

What the card was doing during the window that sets the score:

| | n | mean clock | % of boost | mean temp | temp first → last | power | util |
|---|---|---|---|---|---|---|---|
| cold | 403 | 1475 MHz | 58.5% | 70.6 C | 31 → 66 | 312 W | 86% |
| warm | 406 | 1463 MHz | 58.1% | 73.5 C | 48 → 66 | 313 W | 86% |
| recooled | 399 | 1484 MHz | 58.9% | 69.3 C | 30 → 65 | 311 W | 86% |

**Full range in mean clock across all three thermal states: 1.46%** (1463 to 1484 MHz).
Cold against warm alone is +0.81%.

The three runs order exactly as the physics predicts and the published grade ignores all of
it. Starting temperature 30, 31 and 62 C produces mean scored-window clocks of 1484, 1475
and 1463 MHz and mean temperatures of 69.3, 70.6 and 73.5 C — monotonic in both. So the
effect is real, it is measurable at 1 Hz from outside the product, and it is 1.46% wide. A
quartile letter cannot express 1.46%, which is why all three runs returned D.

Minute by minute inside the scored window:

| minute | cold °C | warm °C | cold MHz | warm MHz |
|---|---|---|---|---|
| 0 | 56.9 | 67.7 | 1120 | 1130 |
| 1 | 68.3 | 73.0 | 1431 | 1404 |
| 2 | 72.3 | 74.6 | 1442 | 1426 |
| 3 | 74.5 | 75.6 | 1491 | 1478 |
| 4 | 75.5 | 75.9 | 1386 | 1395 |

## This retracts a claim I was about to make

The draft memo said: *on a cold L40S, a 2-to-5-minute pass reads 3 to 6% high.* That was
inferred from my own harness, which measured a +8.50% 60-second bias on an L40S. Measured
inside QuickMark on this card, the cold-versus-warm difference is **+0.81%**, and the
published score is identical.

The inference did not transfer, for two structural reasons that are visible in the data
above and that the harness could not have predicted:

1. **The speed pass is long enough to absorb the cold start.** It runs 400 s. The card
   passes 68 C inside the first minute and 75 C by minute three whether it started at 31 C
   or at 62 C. By minute 1 the two runs are within 5 C of each other and by minute 3 within
   1 C. A cold head start that lasts 90 seconds is a small share of a 400-second average.
2. **The agent's own preamble re-cools a hot card.** Between launch and the first compute,
   QuickMark spends roughly 70-90 s on inventory, GPU and network collection and monitoring
   setup, with the GPU idle. The card entered the warm run at 62 C and was at 48 C when
   scoring began. You cannot hand QuickMark a hot GPU; it idles one for you first.

Credit where it is due: **QuickMark already handles warm-up adequately on this card.** A
memo arguing that SiliconMark's test is too short to escape the thermal transient would
have been wrong, and would have been wrong in a way the product owner could disprove in
one run.

## What survives, and is stronger for it

The test-length critique weakens; the *structure* critique does not.

- **Two thirds of the run cannot change the answer.** The thermal probe takes 702 s of the
  1169 s run at 14% mean utilisation. The documentation states: "If that thermal check
  fails, the speed numbers still stand." So 64% of the runtime is spent on a check that is
  advisory, while the 36% that sets the score runs inside the thermal ramp the check exists
  to characterise.
- **The scored window is not at steady state.** The card climbs from 57 C to 75 C during
  the 402 s that sets the score. It happens to land close to its equilibrium by the end,
  which is why cold and warm agree, but that is a property of this card's ramp time against
  this test's duration, not a guarantee. On a card whose ramp is slower than 400 s, or in a
  host that keeps it cooler, the same test would not land in the same place.
- **The letter cannot express the variation.** SM Score is a quartile grade, A to D. A
  grade with four buckets cannot resolve a few percent unless the unit sits on a boundary,
  so "D cold, D warm" is consistent with a real difference that the output simply cannot
  represent. This is the strongest argument for reporting a measured value and an interval
  beside the letter, rather than instead of it.

## The claim that replaces it

Not "the test is too short". Rather: **the same SKU's rating is a host property, and
nothing in the certificate records the host.** Two L40S cards in this study produced
equilibria agreeing to 0.06% while one varied 3.8% across repeats and the other was
perfectly repeatable four times over, the difference being how steadily each host held
temperature. Two H100 NVL cards defaulted to a 310 W power limit against a 400 W maximum,
below NVIDIA's published configurable range, which is invisible in the provider's price
list and absent from its API.

A certificate that records sustained clock as a percentage of boost, the power limit
against the card maximum, the link generation against the card maximum, and the loaded
temperature at which the score was taken would distinguish every case above. Today none of
those appear.

## Caveats

- One card. A single L40S in a single host, graded D, which may be entirely correct for it.
  Nothing here says the score is wrong; it says when the measurement is taken.
- The letter is the only observable without an account, so the cold-versus-warm comparison
  rests on the published grade plus my own telemetry, not on SiliconMark's internal numbers.
- The warm condition was weakened by the agent's own idle preamble, so "warm" here means a
  card at 48 C at first compute, not 62 C. A stronger test would need the soak to continue
  until the moment scoring starts, which the agent's structure does not permit.
- Agent build 2026-09-28. A test-channel build dated 2026-11-04 exists at
  `downloads.silicondata.com/test/agent` and was not the one measured.
