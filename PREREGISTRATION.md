# Pre-registered predictions

Written before the data existed. Git history is the timestamp. Bands are stated in
**percentage points** unless marked relative, so no outcome can be read as a match after
the fact.

## Reference: first L40S (Runpod Secure, uncapped 350 W, boost 2520 MHz)

| quantity | value |
|---|---|
| coolest loaded bin | 1350 MHz = **53.6% of boost** |
| coolest loaded bin, first 5 s of load excluded | 1300 MHz = **51.6% of boost** |
| steady clock | 1163 MHz = 46.2% of boost |
| clock-temperature slope | **-0.171 % of boost per degC**, 95% CI **[-0.181, -0.161]** |
| clock loss after first loaded sample | 7.40 points of boost |
| heat floor (measured, slope excludes zero) | 7.40 points of boost |
| 60 s short-test bias | +8.50% |

Reference: H100 SXM (Runpod Secure, uncapped 700 W, boost 1980 MHz): steady 70.1% of boost,
slope **+0.146 % of boost per degC, CI [-0.096, +0.388], spans zero**, so no measurable
temperature sensitivity. Clock loss after its first loaded sample is 2.61 points, which is
**not** attributed to heat: the slope spans zero, and GEMM throughput fell only 0.91% from
the first 5 s to sustained, so it reads as a power-management transient.

## Prediction 1, second L40S on Runpod Community

The cap portion is tested with the **coolest loaded bin**, not the steady clock. Steady clock
combines cap and heat and cannot isolate either.

- **The power-cap portion is a property of the SKU** if the second card's coolest loaded bin
  falls within **53.6 +/- 2.0 percentage points of boost**, i.e. **51.6% to 55.6%**.
- **Same test, transient excluded (added before the second L40S existed).** The coolest
  loaded bin can catch the first instant of load, before the power controller clamps the
  clock: that is exactly what inflated the H100 figure, where skipping the first 5 s moves
  it from 72.7% to 68.2% of boost. On the first L40S the same exclusion moves it from 53.6%
  to **51.6%**, and is stable at 51.2% when 10 s are excluded. So the cap portion is a
  property of the SKU if the second card's coolest loaded bin **with the first 5 s of load
  excluded** falls within **51.6 +/- 2.0 percentage points**, i.e. **49.6% to 53.6%**.
  Both versions are reported. The original band above is unchanged; this one sits beside it,
  and where they disagree the transient-excluded version is the one to believe.
- **The heat portion is a property of the host** if its slope falls **outside**
  [-0.181, -0.161] % of boost per degC.
- **The heat portion is a property of the SKU** if its slope CI **overlaps**
  [-0.181, -0.161] **and** its measured heat floor is within **7.40 +/- 1.5 percentage
  points**, i.e. **5.9 to 8.9**.
- If the slope CI **spans zero**, this host has no measurable temperature sensitivity and the
  L40S ramp belongs to the first box, not the SKU.

## Prediction 2, cold versus cold-again on the same card

Within one heating curve, temperature and elapsed time rise together, so a single cycle
cannot say which drives the clock. The cooldown repeat supplies a second, independent curve.

- **Temperature is the driver** if the two cycles' slope 95% CIs **overlap** *and* their
  predicted clocks at the shared reference temperature agree within **2% relative**
  (about +/- 23 MHz at 1163 MHz).
- **Something cumulative is at work** if the cold-again cycle predicts a clock more than 2%
  relative **below** the cold cycle at the same temperature. A certificate would then not be
  refreshable by letting the card rest.

## Analysis freeze

The analysis code freezes once the cold-again repeat and the second L40S have been scored.
Remaining runs after that are data collection only. Anything further is appendix material:
the one-page memo carries two lines of method and points here for the rest.
