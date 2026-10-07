# RX 7900 XTX + Ryzen 5 7600X on Linux: which settings beat the defaults

Measured 2026-10-06 and 2026-10-07 on one machine, Flash-Next IQ3_S, engine built from source (HIP). Strata's
PC-dependent defaults were measured on a Ryzen 5 7600 with an RTX 5070 ([DETAILS.md](../../../docs/DETAILS.md)) and
setup does not offer calibration on AMD yet ([AMD_HIP.md](../../../docs/AMD_HIP.md)), so these are A/B sweeps of
engine arguments against the setup-written configuration on this PC. Every run is in [runs.csv](runs.csv).

**Result for this PC:** `--kv-resident 32768`, `--prefill auto:32768` and `--pcie-frac 0.35` on top of setup's
configuration. Against setup's configuration: decode +9% (65.5 to 71.6 tok/s on prose), a 53.7k-token prompt read
21% faster (1,443 to 1,750 tok/s), the same free VRAM. Everything else tried was neutral or worse.

## Hardware and software

- **GPU:** AMD Radeon RX 7900 XTX 24 GB (gfx1100), PCIe 4.0 x16 (link 16 GT/s x16), Resizable BAR on (32 GiB
  BAR0), power cap 305 W (driver default). The desktop (GNOME, Wayland) runs on the same card and keeps ~0.9 GiB.
- **CPU:** AMD Ryzen 5 7600X, 6 cores / 12 threads, AVX-512 (F, VL, VNNI, BF16); `amd-pstate-epp`, governor and
  EPP `performance`.
- **RAM:** 64 GB DDR5 (2 x 32 GB) at 5800 MT/s, 60 GiB visible to Linux; transparent huge pages `madvise`.
- **Storage:** model files on an NVMe SSD (Corsair MP600 CORE XT).
- **OS:** Ubuntu 26.04.1 LTS, kernel 7.0.0-34-generic, in-kernel amdgpu.
- **ROCm:** TheRock wheels 7.10.0a20251120 in Strata's venv (setup's choice for gfx110X), no system ROCm.
  hipBLASLt tuning table `tools/hip/gfx1100-hipblaslt-100200.txt`, `STRATA_HIP_WMMA=1`.
- **Engine:** source build with setup's options (`-DSTRATA_ENABLE_HIP=ON -DSTRATA_PREFILL_MMQ=ON`, gfx1100,
  llama.cpp pin `3cf0325`, ggml `GGML_NATIVE=ON`, so `-march=native`). Sweep A: engine 0.1.40 at `829704a`.
  Sweep B and the image test: 0.1.40.2 at `de78427`. Both are this fork's `ash/main` (see [FORK.md](../../../FORK.md)).

## Model and base configuration

- **Model:** `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, IQ3_S
  (`Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-0000{1,2}-of-00002.gguf`), native pack prepared by setup (`packs/iq3_s`),
  shipped `data/expert-profile.bin`, MTP draft layer from setup (`mtp/rt`). No calibration, no speed projection.
- **Setup's arguments (the baseline):** `--expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5
  --max-context 131072 --kv int8`, plus `--vram-reserve-mib 5120` (raised from setup's value so the desktop keeps
  at least ~3.4 GiB of VRAM free while the model works: the compositor crashes if a model starves it).
- **Server:** one request at a time, no conversation cache, images off except in the image test.

## Workload

[bench_client.py](bench_client.py) against the server on a free port, one fresh server per run (each arm's numbers
include no state from another), model loaded before timing:

1. three greedy prompts (answers compared across arms);
2. a seeded warm-up, then two seeded 1,200-token generations (prose essays) -> **decode**, tokens per second of the
   engine's own timing, reasoning included (`reasoning_effort: low`);
3. a 53,715-token prompt (llama.cpp source) + a 400-token answer -> **long prompt** read speed and decode at that
   depth, then a ~2,450-token **follow-up** turn on the same conversation (the 53.7k prefix reused).

Sampling for 2 and 3: temperature 0.8, top_k 40, top_p 0.95, min_p 0.05, sent with each request.
[bench_tune.py](bench_tune.py) adds an **agent-style edit** (sweep B): ~9,000 characters of C++ in the prompt,
re-emitted whole with one function renamed, twice (temperature 0.6, top_k 20, top_p 0.95, seeded) - most of the
answer is a copy of the context, as with a coding agent's edit tool. Every edit was checked for the rename.
A watchdog stopped any run below 1.5 GiB free VRAM or 2 GiB available RAM; none was stopped.

**Noise:** the first prose generation is very repeatable between server starts (68.9-69.3 tok/s on the baseline
in five runs); the second varies more (63.6-65.8), the 400-token answer at depth by +-3 tok/s, the edit test by
+-5% between server starts. Arms were interleaved with repeated baselines; the tables give the median and range of
all runs of an arm.

## Sweep A (engine 0.1.40): KV streaming and the prompt chunk

Two arms each, interleaved. `--kv-resident 32768` keeps 32,768 cells of each attention layer's K/V in VRAM and the
rest in pinned RAM; the VRAM it frees goes to expert slots. `--prefill auto:32768` lets `auto` read prompts in
chunks above 8,192 tokens (it chose 25,344-28,160).

| arm | decode tok/s | 53.7k prompt tok/s | decode at 53.7k | expert slots | min free VRAM | min available RAM |
|---|---|---|---|---|---|---|
| setup's | 65.5 [65.3-66.0] | 1,443 [1,441-1,445] | 67.4 [65.2-69.6] | 6,141 | 3,377 MiB | 6.3 GiB |
| `--kv-resident 32768` | 69.2 [66.2-70.4] | 1,438 | 68.6 [65.9-71.2] | 6,812 | 3,376 MiB | 4.6 GiB |
| `--prefill auto:32768` | 65.3 [64.2-65.9] | 1,680 [1,679-1,682] | 70.0 [68.0-72.1] | 6,141 | 3,378 MiB | 6.1 GiB |
| **both** | **68.2 [66.0-70.4]** | **1,755 [1,755-1,755]** | 68.2 [66.1-70.4] | 6,812 | 3,375 MiB | 4.4 GiB |

At **109,233 prompt tokens** (two runs each): both read the prompt at 1,683 [1,680-1,687] tok/s against 1,387
[1,385-1,388] (+21%, 71 s instead of 85 s), and decoded the 400-token answer at 64.7 [63.5-65.8] against 67.6
[65.1-70.1] (-4%, inside that measurement's spread but in the expected direction: past 32,768 cells the K/V streams
from RAM). The pinned K/V costs ~1.8 GiB of available RAM at 131,072 context. The greedy answers changed in one of
three (the last clause of one sentence) with `--kv-resident`: more experts are computed by the GPU, which rounds
differently from the CPU; each configuration repeats its own answers exactly.

## Sweep B (engine 0.1.40.2): the PC-dependent settings

Base = setup's arguments + `--kv-resident 32768 --prefill auto:32768`. Phase 1: one run of each arm, the base three
times; phase 2: the candidates twice each, interleaved, with the base twice.

| arm | runs | decode tok/s | decode at 53.7k | edit tok/s | verdict |
|---|---|---|---|---|---|
| base | 5 | 67.3 [63.6-69.3] | 68.8 [65.1-69.3] | 87.2 [81.3-91.1] | |
| **`--pcie-frac 0.35`** | 3 | **71.6 [66.9-74.3]** | **72.2 [72.0-72.3]** | **92.7 [88.0-95.4]** | **+6% decode** |
| `--pcie-frac 0.25` | 2 | 68.2 [66.7-70.5] | 69.2 [68.1-70.3] | 89.8 [88.7-91.2] | less than 0.35 |
| `--pcie-frac 0.75` | 1 | 61.6 [59.6-63.6] | 62.9 | 77.3 [77.3-77.4] | -8% |
| `--host-core last` | 1 | 67.0 [64.8-69.2] | 69.4 | 89.8 [89.0-90.7] | neutral |
| 0.35 + `--host-core last` | 2 | 71.3 [68.4-74.1] | 71.5 [70.8-72.2] | 93.0 [90.5-94.2] | = 0.35 |
| `--lookup-chain 2` | 1 | 66.3 [65.7-66.9] | 66.5 | 90.7 [89.6-91.7] | |
| 0.35 + `--lookup-chain 2` | 2 | 69.6 [66.1-72.2] | 73.9 [73.0-74.8] | 95.8 [94.2-98.3] | edits +4%, prose -2.5% vs 0.35 |
| `--lookup-chain 4` | 1 | 67.5 [65.7-69.3] | 65.2 | 89.5 [89.4-89.6] | = chain 2 |
| `--pool-workers 4` (default 5) | 1 | 65.5 [62.1-68.8] | 65.9 | 88.3 [86.9-89.8] | neutral |
| `--spec-min-p 0.3` | 1 | 63.3 [61.6-65.0] | 70.1 | 82.7 [82.0-83.3] | neutral or worse |
| `--spec-min-p 0.7` | 1 | 68.0 [67.4-68.5] | 66.4 | 88.8 [87.5-90.1] | neutral |
| 0.35 + GPU `COMPUTE` power profile | 1 | 71.6 [68.9-74.3] | 71.8 | 94.2 [93.7-94.7] | = 0.35 |

- **`--pcie-frac`** (the share of the experts missing from VRAM that are copied to the GPU rather than computed by
  the CPU) is the one setting that pays here. The engine's probe measured 28.6 GB/s host to device and picked 0.55;
  on this CPU 0.35 is the fastest value for decode, decode at depth and edits (prompt reading is unchanged), and the
  first prose generation, the most repeatable number, reads
  74.2-74.3 tok/s in all three of its runs against 68.9-69.3 in all five of the base's. 0.25 and 0.75 are both
  slower. Prompt read speed is unchanged (1,750 tok/s). One greedy answer differs from the base's (a variable name
  in the palindrome function; both correct), the same way in all three runs.
- **`--host-core last`:** on this PC the GPU's interrupt (amdgpu, IRQ 115) is handled on logical CPU 6, the SMT
  sibling of physical core 0 where the host loop spins by default (`irqbalance` is not running). Moving the host to
  the last core measured neutral.
- **`--lookup-chain`** speeds up the edit test (copied code) and slows prose by the rejected drafts; for a coding
  agent whose output is mostly reasoning text it does not pay on balance. With the default suffix drafts the edit
  test already runs at ~87 tok/s against ~67 on prose.
- **GPU power profile:** `pp_power_profile_mode` 5 (COMPUTE, with manual DPM) instead of the default 0 measured the
  same as the default; the card already runs its high clocks under this load.
- Unchanged by any arm: the 53.7k prompt's read speed (1,713-1,756 tok/s), the follow-up's (818-878 tok/s), the free
  VRAM (no run below 3,374 MiB), the expert slots (6,804-6,805).

## Images through the CPU encoder

The CPU image encoder (`--vision cpu`, `strata-vision` built from `tools/vision` against the pinned llama.cpp
without CUDA/HIP, 6 threads) with the F16 mmproj of the same model
(`mmproj-Qwen3.8-Flash-Next-F16.gguf`; setup downloads the BF16 one), `max_tokens` 300, on engine 0.1.40.2 with the
sweep B base arguments: a 1280x800 PNG of a mock settings page ([vision-mock-ui.png](vision-mock-ui.png)) with two
planted layout faults (a button overlapping a text field, an error line colliding with the next label) and a
greyed-out button and the question "describe every piece of text you can read, the colours of the buttons, and any layout or
UI problems you can see" (temperature 0.6, `reasoning_effort: low`).

- The prompt was 352 tokens (the image ~300) and was read in 1.4 s, encoding included; 1,124 tokens of answer at
  72.5 tok/s; 19.3 s in all.
- The answer read every text on the page, named both layout faults and the button colours, and called the
  greyed-out button disabled.
- Cost: 6,804 expert slots against 6,805 without images, free VRAM unchanged (min 3,534 MiB), at least 4.6 GiB of
  RAM still available. On this PC images cost the GPU nothing.

## Limits

One machine, one model. Two to five runs per arm, one for the phase 1 arms that were dropped. The decode numbers come
from the engine's timing with MTP drafts and suffix drafts on, so they depend on the text: prose essays and copied
code here, not a measured agent session. The prompts are built from a llama.cpp checkout at `2e7c58c54` plus local
patches (any recent llama.cpp source gives prompts of similar length and kind). The edit test's +-5% spread between
server starts is larger than most of the effects in it.
