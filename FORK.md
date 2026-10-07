# This fork

A personal fork of [Niko1221/Strata](https://github.com/Niko1221/Strata) that keeps track of the upstream pull
requests worth having on one machine: an RX 7900 XTX 24 GB with a Ryzen 5 7600X and 64 GB of RAM on Ubuntu, running
Qwen3.8-Flash-Next IQ3_S through the HIP backend. Measurements:
[bench/results/2026-10-07-rx-7900-xtx-linux-tuning](bench/results/2026-10-07-rx-7900-xtx-linux-tuning/README.md).

## Branches

- **`ash/main`** - upstream `main` plus the pull requests below. This is what that machine runs.
- `main` is not used; upstream is the `upstream` remote.

A pull request goes into `ash/main` only after an A/B on that machine shows it helps (speed, or a fix that changes
what the machine sees). Pull requests that upstream supersedes are not merged here.

## What `ash/main` carries on top of upstream

| change | what | why |
|---|---|---|
| upstream `main` | 0.1.40.2 (`e8ca9af`), merged in `de78427` | |
| [#1126](https://github.com/Niko1221/Strata/pull/1126) | server, installer and engine hardening | security |
| `829704a` | #1126's `index.txt` shape guard accepts `ne1 == 0` | without it the engine refuses every native pack ("index.txt: bad shape for output_hc_norm.weight"): `tools/iq_pack.py` writes `ne1 = 0` for 1-D tensors |
| [#1107](https://github.com/Niko1221/Strata/pull/1107) | prefill: grouped gather on short prompts too | in the build that won the 0.1.40 A/B |
| merge `de78427` | #1107 against upstream's #789 (`src/prefill/prefill.cpp`) | #789 stages experts by in-flight transfer count, which assumes each compute records its slot's release; #1107's grouped walk records the release at the group's flush, so it keeps its own position bound and #789's lookahead applies when not grouped |

[#1240](https://github.com/Niko1221/Strata/pull/1240) was carried until upstream took it as `ac52c2c`.

## Tested and not merged

| pull request | result on that machine |
|---|---|
| [#1125](https://github.com/Niko1221/Strata/pull/1125) AVX2 Q8_K quantizer | no measurable change |
| [#1182](https://github.com/Niko1221/Strata/pull/1182) cancellation during quiet prefill | a cancelled 53.7k prompt freed the server in 5.8 s with and without it |
| [#1181](https://github.com/Niko1221/Strata/pull/1181) verify-window plan in O(n) | no measurable change |
| [#1368](https://github.com/Niko1221/Strata/pull/1368) quantize once and scatter | no measurable change |
| [#1296](https://github.com/Niko1221/Strata/pull/1296) GDN WY recurrence | its WMMA arm needs rocWMMA, which the TheRock wheels lack; the FMA fallback reads prompts at 135 instead of ~1,750 tok/s |
| [#1362](https://github.com/Niko1221/Strata/pull/1362) tool-call newline tracking | built and ran in the A/B; its tests were not run and no case was seen on that machine |
| [#1316](https://github.com/Niko1221/Strata/pull/1316), [#1298](https://github.com/Niko1221/Strata/pull/1298) | conflict with 0.1.40.2; not retried |

## Updating

1. `git fetch upstream`, then merge `upstream/main` into a copy of `ash/main` in its own worktree.
2. Build it with the same options as the running engine and A/B it against the running one (same configuration,
   interleaved runs, greedy answers compared).
3. If it is no slower and its answers hold, fast-forward `ash/main` to it, push, and point the service at it.
