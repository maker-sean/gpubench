# Three L40S cards, three hosts: what the host sets and what it does not

Three NVIDIA L40S cards, three different hosts on the same provider and tier, same 350 W
power limit on every one, same harness, same bf16 GEMM, 600 s or 1200 s per repeat. Steady
clock and steady temperature are the mean over the final quarter of each repeat, taken from
the raw 500 ms telemetry.

| card | steady temp, across repeats | temp spread | sustained TFLOPS | repeat spread | cold 60 s bias |
|---|---|---|---|---|---|
| A | 72.2–79.3 °C | **7.1 °C** | 163.6–169.9 | **3.85%** | +8.50% |
| B | 68.0–68.0 °C | **0.0 °C** | 163.4–163.5 | **0.06%** | +2.71% |
| C | 79.3–80.0 °C | **0.7 °C** | 157.1–158.2 | **0.70%** | +8.71% |

Four findings fall out, and they are not the same finding.

## 1. Per clock, the three cards are identical. Everything else is clock.

| card | sustained TFLOPS | steady SM clock | TFLOPS per MHz |
|---|---|---|---|
| A | 163.6 | 1163 MHz | **0.1407** |
| B | 163.5 | 1162 MHz | **0.1407** |
| C | 158.2 | 1125 MHz | **0.1406** |

Identical to four significant figures. Card C delivers 3.3% less than card A for exactly one
reason: it holds 1125 MHz instead of 1163. 1125/1163 is 96.7%, and 158.2/163.6 is 96.7%.

This is worth stating first because it says what a rating actually measures. No card in this
study did less work per clock than another of its model. The spread between hosts is a clock
measurement and nothing else, and clock is set by the power limit and the temperature — one
of which is configuration and the other of which belongs to the building.

The same holds across the H100 variants: the NVL delivers 0.4996 TFLOPS/MHz against the
SXM's 0.4982, or **100.3% of the SXM's work per clock**, while delivering 49% of its
throughput. Measured against measured, no specification sheet required.

## 2. Temperature sensitivity is similar in magnitude across cards, but not the same number

All three cards had a cold repeat whose loaded temperature span cleared the 15 °C minimum
this study requires before reporting a slope:

| card | slope, cold repeat | 95% CI | loaded span |
|---|---|---|---|
| A | **−0.171 %/°C** of boost | [−0.180, −0.161] | 35 °C (45–80) |
| B | **−0.205 %/°C** of boost | [−0.241, −0.169] | 17 °C (51–68) |
| C | **−0.217 %/°C** of boost | [−0.229, −0.205] | 30 °C (50–80) |

Every card loses roughly a fifth of a percent of boost per degree, so the effect is real and
its order of magnitude is a property of the model. But the two cards measured over the widest
spans, A and C, have **non-overlapping confidence intervals** — 27% apart in slope. Card B's
interval is wide enough to overlap both, which is what a 17 °C span buys.

So there is no single L40S number to apply. A temperature correction has to come from the
card's own history, not from its model. An earlier revision of this document claimed the
slopes agreed and reported only two of the three; that was wrong, and it was wrong because
card A — the one that breaks it — was dropped from the table by a bad join.

## 3. Across cards, temperature predicts nothing at all

- A and C both settle at **79.3 °C** and differ by **3.3%** (163.6 vs 158.2 on the cold repeat).
- A and B are **11.3 °C apart** and deliver the **same throughput** (163.6 vs 163.5, 0.06%).

Applied across units, even the shallowest of the three slopes predicts about 1.9% for an
11 °C gap; observed is 0.06% for that pair and 3.3% for a pair with no gap at all. Knowing a
card's operating temperature tells you nothing about its score.

This is the error the obvious analysis makes. A within-unit slope with a tight confidence
interval invites extrapolation across units, and on this data that extrapolation is wrong
twice over: wrong in magnitude for one pair and wrong in direction of effect for the other.

## 4. The host contributes two independent things, and they are easy to confuse

**How hot the host runs sets the cold-start bias.** The two cards that equilibrate at 79.3 °C
read +8.50% and +8.71% high on a 60-second test — agreeing to a fifth of a point. The card
that equilibrates at 68.0 °C reads +2.71%. The ramp a short test misses is the ramp to the
host's own equilibrium, so a hot box has three times the cold bias of a cool one.

**How steadily the host holds that temperature sets the repeatability.**

| card | steady temp moved | rating varied |
|---|---|---|
| A | 7.1 °C | 3.85% |
| B | 0.0 °C | 0.06% |
| C | 0.7 °C | 0.70% |

Monotonic across all three. The card in the box that held temperature constant returned the
same rating four times to within 0.06%, including after a ten-minute cooldown. The card in
the box that let temperature wander 7 °C produced a 3.85% spread — larger than the 1.50%
spread this study measured between two different H100 SXM cards.

These two properties are independent, which is the trap. Card C is the steadiest host in the
set (0.7 °C, 0.70% spread) and also has the **largest** cold bias of any card measured
(+8.71%), because it runs hottest. A host that gives you a highly reproducible number can
still give you a badly biased one if the test is short, and a stability metric will not catch
it.

## What this means for a certificate

Four findings, four different fields, and a score alone carries none of them:

- Because the spread between hosts is entirely clock, the **sustained clock as a percentage
  of boost** is the one number that explains a rating, and it is cheap to record.
- Because sensitivity varies by unit, the **loaded temperature at which the score was taken**
  must be recorded, and **no cross-unit temperature adjustment should be offered**.
- Because equilibrium temperature sets the cold bias, a certificate needs to show the test
  reached steady state — not merely that it ran for a fixed time.
- Because repeatability is a host property, the **spread across repeats** belongs beside the
  score. A single number from a 7 °C-wandering host and a single number from a steady one
  look identical on paper and are not the same claim.

A fleet percentile such as an SM Score inherits all of this: it ranks units partly by the
hosts they sat in. Card C, the slowest of the three, graded **D** — the bottom quartile —
while running within 0.7 °C of perfectly steady. Its repeatability was excellent, its work
per clock was identical to its siblings', and its rating was still 3.3% below card A in a
different box.

## Caveats

- Three cards. The monotonic relationship between temperature steadiness and rating spread
  is based on three points and should be read as a consistent pattern, not a fitted law.
- Host assignment is not controllable on this provider, so "host" here bundles cooling,
  airflow, ambient temperature and neighbours into one unobserved variable.
- Card A's four repeats include one at 169.9 TFLOPS and 72.2 °C that is the outlier driving
  its 3.85% spread; the other three agree to 0.2%. The spread is reported as the range
  because a certificate would have had to quote whichever repeat it happened to run.
- Card C was still drifting slowly downward at the end of its two warm repeats (−0.14%/min
  and −0.018%/min, both flagged by the harness's own settling check as "do not rate"). Its
  figures are reported with that flag attached rather than silently averaged.
- Cards A and B ran 1200 s repeats; card C ran 600 s. Steady-state figures are taken from the
  final quarter of each, so they are comparable, but card C had less time to drift.
