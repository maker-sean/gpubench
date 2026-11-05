# Three cards called "H100", three different answers to "which is cheapest"

All three were rented from the same provider (Runpod Secure), on the same day, with the
same harness at the same git revision. All three appear in the provider's catalogue under
the name **H100**. The price list distinguishes them only by VRAM, RAM, vCPU and price.

Measured: bf16 GEMM sustained for 600 s (median of 1-second throughput windows, cold
repeat), memory bandwidth, and — on two of the three — vLLM serving Qwen2.5-7B-Instruct,
256 prompts x 256 output tokens, median of 3 repeats.

## What the cards are

| | $/hr | boost | power limit / default / max | PCIe link | GEMM TFLOPS | mem GB/s | steady MHz | W | T |
|---|---|---|---|---|---|---|---|---|---|
| H100 SXM | 3.99 | 1980 | 700 / 700 / 700 | gen5/5 x16 | **682.0** | 3019 | 1369 | 697 | 72 |
| H100 NVL | 3.19 | 1785 | **310** / 310 / 400 | gen5/5 x16 | 334.7 | **3542** | 670 | 309 | 52 |
| H100 PCIe | 2.89 | 1755 | **310** / 310 / 350 | **gen4**/4 x16 | 349.8 | **1851** | 792 | 309 | 54 |

Percentage of boost clock sustained: SXM 69%, NVL 38%, PCIe 45%.

Per clock, the SXM and the NVL are the same card: 0.4982 against 0.4996 TFLOPS per MHz, or
**100.3%** of the SXM's work per clock, while the NVL delivers 49% of its throughput. The
NVL's entire shortfall is clock, and its clock is set by the 310 W limit. The PCIe part sits
at 0.4417 TFLOPS/MHz, 88.7% of the SXM, consistent with its smaller SM count rather than
with any configuration. No published specification is needed for either comparison; these
are measured numbers divided by measured clocks.

All three sat at 100% of their configured power limit for the whole run. None was
thermally throttled. Every one of them is power-bound, and two of them are power-bound at
a limit well below what the card's own hardware permits.

Two things here are invisible at purchase and are not in the price list:

- **The NVL ships configured at 310 W against a 400 W maximum.** NVIDIA publishes a
  350-400 W configurable range for this card, so the configured limit is below the range the
  vendor documents. Both NVL cards reported `power.limit` equal to `power.default_limit` at
  310 W, so this is the state the card arrived in rather than a limit a host lowered at boot
  — which makes it harder to find, not easier, since there is no deviation from default to
  detect. Datacenter not recorded.
- **The PCIe card is on a gen 4 link.** Its own `pcie.link.gen.max` reads 4, so this is a
  gen-4 part, but its measured bandwidth is 52% of the NVL's at a similar compute number.

## The same name, three answers

| | $/1k TFLOP-h | $/1k GB/s-h | M output-tok per $ | rel. compute | rel. bandwidth | rel. tokens |
|---|---|---|---|---|---|---|
| H100 SXM | 5.85 | 1.322 | 24.09 | 1.00x | 1.00x | 1.00x |
| H100 NVL | 9.53 | **0.901** | 16.61 | 1.63x | **0.68x** | 1.45x |
| H100 PCIe | 8.26 | 1.562 | not run | 1.41x | 1.18x | - |

Relative columns are cost per unit of delivered work, SXM = 1.00, lower is cheaper.

Read the rankings in order and they disagree with each other:

- **Per unit of compute**, the most expensive card is the cheapest: SXM at 1.00x, PCIe
  1.41x, NVL 1.63x.
- **Per unit of bandwidth**, the ranking inverts at the top: the NVL is 32% cheaper than
  the SXM, and the PCIe card is the worst of the three.
- **Per output token**, the ranking inverts back: the NVL costs 45% more per token than
  the SXM.

A buyer who picks on sticker price takes the PCIe card, which has the least bandwidth of
the three by a wide margin — 52% of the NVL's — while sitting between the other two on cost
per unit of compute. A buyer who picks on bandwidth takes the NVL and pays 45% more per
token than if they had bought the card with the higher sticker price. No single axis ranks
these three the same way, which is the point.

## Why the bandwidth bargain does not survive contact with a real workload

| | out tok/s | SM clock | % of boost | power | % of its own cap | power-capped | model load |
|---|---|---|---|---|---|---|---|
| H100 SXM | 26,700 | 1838 MHz | 93% | 522 W | 75% of 700 W | 56% of samples | 139 s |
| H100 NVL | 14,717 | 956 MHz | 54% | 290 W | **94% of 310 W** | **92% of samples** | 251 s |

The NVL has 17% more memory bandwidth than the SXM and costs 20% less, which is why it
looks like the better buy per GB/s. Under vLLM it delivers 55% of the SXM's tokens per
second.

The reason is in the last three columns. The SXM meets this workload with a quarter of its
power budget unused and holds 93% of its boost clock. The NVL is against its 310 W ceiling
for 92% of the run and gives up almost half its clock to stay there. Bandwidth it has;
clock to use the bandwidth it does not.

So the census finding and the economics finding are one finding. The 310 W default is not
a specification footnote — it is the mechanism that turns a 32% bandwidth discount into a
45% token premium. It does not appear in the provider's price list, in the provider's API
(10 fields, none of them a power limit), or anywhere a buyer looks before renting.

## What this implies for a benchmark product

A single score cannot rank these three cards, because the correct ranking depends on the
buyer's workload and the three rankings here are mutually inconsistent. What a certificate
can do is record the configuration that produced the number — power limit against card
maximum, link generation against card maximum, and the clock actually sustained as a
percentage of boost. Each of the three cards above would then be distinguishable on paper
before anyone rents one, which today they are not.

## Caveats

- One card of each variant. These are host-and-configuration results, not SKU results, and
  a second NVL could well be configured at 400 W.
- The inference test was run on two of the three variants; the PCIe card was acquired late
  and only received the GEMM and bandwidth pass.
- The inference figures come from a different physical card of the same variant than the
  GEMM and bandwidth figures above (a second SXM and a second NVL, both at the same power
  limit as the first: 700 W and 310 W respectively). So the compute and token numbers are
  variant-level results, not two measurements of one unit, and the 45% token premium rests
  on the variants sharing a configuration rather than on a single card doing both tests.
- Model load time (139 s vs 251 s) is a host storage and network property, not a GPU
  property, and is reported only to show it was not confused with serving throughput.
- Tokens per dollar uses on-demand hourly price and excludes model load time, which a
  buyer running a persistent server pays once.
