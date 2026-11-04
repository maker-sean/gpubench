# Three L40S cards, three hosts: what the host sets and what it does not

Three NVIDIA L40S cards, three different hosts on the same provider and tier, same 350 W
power limit on every one, same harness, same bf16 GEMM, 600 s per repeat. Steady clock is
the last-quarter mean; steady temperature is the end-of-run value.

| card | steady temps | temp spread | TFLOPS | repeat spread | mean |
|---|---|---|---|---|---|
| A | 73–79 °C | 6 °C | 163.6–169.8 | **3.78%** | 165.4 |
| B | 68 °C | 0 °C | 163.4–163.5 | **0.08%** | 163.4 |
| C | 79–80 °C | 1 °C | 157.1–158.3 | **0.72%** | 157.9 |

Three findings fall out, and they are not the same finding.

## 1. Temperature sensitivity is a SKU property

Two of the three cards had a loaded temperature span wide enough to regress on (the 15 °C
minimum this study applies), and they agree:

| card | slope | 95% CI | loaded span |
|---|---|---|---|
| B | **−0.205 %/°C** of boost | [−0.241, −0.169] | 17 °C |
| C | **−0.217 %/°C** of boost | [−0.229, −0.205] | 30 °C |

Different cards, different hosts, overlapping intervals. Within a card, clock falls with
temperature at about a fifth of a percent per degree, and that number looks like a property
of the L40S rather than of any one unit.

## 2. Across cards, temperature predicts nothing

The same slope that is solid within a card has no cross-card power at all:

- A and C both settle at **79 °C** and differ by **3.3%** (163.6 vs 158.2 on the cold repeat).
- A and B are **11 °C apart** and deliver the **same throughput** (163.6 vs 163.5, 0.06%).

Applied across units the slope predicts 2.3% for an 11 °C gap; observed is 0.06% for that
pair and 3.3% for a pair with no gap at all. So a card's rating is set by something other
than the temperature it runs at — unit-to-unit variation, host configuration, or both — and
knowing a card's operating temperature does not let you infer its score.

This is worth stating plainly because it is the error the obvious analysis makes. A
within-unit slope with a tight confidence interval invites extrapolation across units, and
on this data that extrapolation is wrong twice over: wrong in magnitude for one pair and
wrong in sign-of-effect for the other.

## 3. Repeatability is a host property

What the host does control is whether the number is reproducible:

| card | steady temp moved | rating varied |
|---|---|---|
| A | 6 °C | 3.78% |
| B | 0 °C | 0.08% |
| C | 1 °C | 0.72% |

Monotonic across all three. The card in the box that held temperature constant returned the
same rating four times to within 0.08%, including after a ten-minute cooldown. The card in
the box that let temperature wander 6 °C produced a 3.8% spread — larger than the 1.5%
spread this study measured between two different H100s.

A card in a thermally unstable host is not certifiable to better than a few percent no
matter how long the test runs, because the variance is in the host, not in the measurement
window.

## What this means for a certificate

The three findings point at three different fields, and a score alone carries none of them:

- Because sensitivity is a SKU property, the **loaded temperature at which the score was
  taken** must be recorded, or the number cannot be compared to another unit's.
- Because temperature does not predict across units, **every unit has to be measured**. A
  model-level rating adjusted for temperature will not reproduce what a buyer rents.
- Because repeatability is a host property, the **spread across repeats** belongs on the
  certificate beside the score. A single number from a 6 °C-wandering host and a single
  number from a stable one look identical on paper and are not the same claim.

A fleet percentile such as an SM Score inherits all of this: it ranks units partly by the
hosts they sat in. Card C, the slowest of the three, graded **D** — the bottom quartile —
while running within 1 °C of perfectly steady. Its repeatability was excellent and its
rating was still 3.5% below a sibling card in a cooler box.

## Caveats

- Three cards. The monotonic relationship between temperature steadiness and rating spread
  is based on three points and should be read as a consistent pattern, not a fitted law.
- Host assignment is not controllable on this provider, so "host" here bundles cooling,
  airflow, ambient temperature and neighbours into one unobserved variable.
- Card A's repeats ran under an older harness revision that did not record last-quarter
  clock, so its steady clocks are absent from the table; its temperatures and throughput
  are directly comparable.
- Card C was still drifting slowly downward at the end of two repeats (−0.14%/min and
  −0.018%/min, both flagged by the harness's own settling check as "do not rate"). Its
  figures are reported with that flag attached rather than silently averaged.
