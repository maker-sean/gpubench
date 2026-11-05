# Not All GPU Hours Are Created Equal

Sean Mihm · November 2026 · [PDF](memo/Not-all-GPU-hours-are-created-equal.pdf)

Silicon Data's indices rest on a rated GPU hour being a known, interchangeable quantity. I
spent two days renting and measuring GPUs to test that, and it breaks on one variable: the
host the card runs in. Closing that gap lets a rating carry more weight when settling a
futures contract, deciding an RMA claim, or backing a loan.

## What the research showed

Three things decide how much work a GPU actually delivers, and neither its model name nor
its grade specifies them.

**1. "H100" ships in different power envelopes, and you can't see which before renting.**
On Runpod, six H100 SXM cards ran at 700 W, while both H100 NVL cards I rented arrived at
310 W, below NVIDIA's published 350 to 400 W range. Runpod, Vast.ai, and Lambda show no
power limit before rental. At that setting, the NVL costs 45% more per output token than the
SXM, despite having more memory bandwidth.

**2. The host's temperature determines how accurately a short test predicts sustained
performance.** Ratings come from short tests, because benchmarking time is time a GPU isn't
earning. On three L40S cards, a 60-second test read about 8.6% high on two hosts that
settled near 79 °C, and 2.7% high on one at 68 °C. H100s stayed under 1%.

**3. The host's temperature stability determines whether a rating repeats.** An L40S in a
steady host returned the same result four times, within 0.06%. One in a host that drifted
7 °C spread 3.85%, more than double the gap between two different H100s. On an unstable
host, a retest can't tell a worse card from a warmer hour.

Per clock, cards of the same chip matched within 0.1%, so every difference above comes from
the host.

## What a rating should record

QuickMark already records power and temperature, so most of this is a display change. The
grade stays; these fields tell a buyer how far to trust it.

| Field | Why it matters | Who it protects |
|---|---|---|
| Power limit and its default | Two "H100s" can differ 2.26x in power and 45% in cost per token | Futures contracts that treat one H100 hour as interchangeable |
| Loaded temperature | Sets how far a short test overstates | Teams validating clusters before turn-up; procurement teams in RMA disputes |
| Repeat spread | Shows whether the number would come out the same twice | Lenders marking collateral; resellers grading inventory |

## Suggested changes

**1. Show the number and conditions behind each grade.** Publish the measured value and its
range beside the letter, plus the test temperature and power limit. A buyer can then tell
apart two cards with the same grade, and when a retest comes in lower, see whether the card
or the host changed. QuickMark already records power and temperature, so this is mostly a
display change.

**2. Price delivered work, not just the card name.** Joining SiliconMark results to rental
records would let buyers compare offers per unit of work delivered, and give contracts a
clear standard for what an H100 hour must provide.

**3. Start a drift study on reserved devices.** The open question is how long a certificate
holds once a card changes hands. At Catalog, I designed an accelerated decay study to answer
the same question for DNA storage and computation.

## Experiment details

I benchmarked three L40S cards and three H100 variants on Runpod, sampling clocks, power,
and temperature every 500 ms, and checked power limits on nine H100s. Predictions were
committed to git before the runs that tested them.

QuickMark graded one L40S D whether started cold, warm, or recooled, so it handles a cold
start well. Open question: grades are fleet quartiles, and my three L40S cards spanned 8%,
so if a quartile is narrower than about 3%, host conditions alone can move a card a full
letter.

Limits: one provider, a few cards per model, hours of testing.

Code and raw data: [github.com/maker-sean/gpubench](https://github.com/maker-sean/gpubench)

---

## Where each number comes from

| Memo claim | Backing |
|---|---|
| Six SXM at 700 W, two NVL at 310 W / 400 W max, one PCIe at 310 W / 350 W max | [VARIANTS.md](VARIANTS.md), power-limit census of nine distinct GPUs by UUID |
| NVL 45% more per output token | [VARIANTS.md](VARIANTS.md), vLLM on Qwen2.5-7B-Instruct |
| 60 s bias 8.50% / 8.71% / 2.71% on three L40S; H100s 0.33–0.70% | [HOSTS.md](HOSTS.md) |
| Repeat spread 3.85% / 0.70% / 0.06%; 1.50% between two SXM cards | [HOSTS.md](HOSTS.md) |
| Per clock within 0.1% (0.1407 / 0.1407 / 0.1406 TFLOPS per MHz) | [HOSTS.md](HOSTS.md) |
| QuickMark D cold, warm and recooled; 1.46% clock range across the three | [QUICKMARK.md](QUICKMARK.md), traces in `data/quickmark/` |
| Grades are fleet quartiles | SiliconMark Benchmarks documentation |
| Required test length: H100s 61.5–61.8 s, L40S 99.4–511.2 s | [CONVERGENCE.md](CONVERGENCE.md) |
| Predictions committed before the runs | [PREREGISTRATION.md](PREREGISTRATION.md) |
