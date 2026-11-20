# Final scorecard: every adopted opt-in port on vLLM 0.34 (config F) against our B70 stack (2026-11-20)

Branch `b70/combined-final` = `b70/vllm-034` (10e9072) + every adopted opt-in port, one linear history (81 commits up
to 7bbcff2, plus this doc). Every knob is still off by default. With all knobs unset, `launch.sh` prints a command
byte-identical to `b70/vllm-034`'s. `test_launch_profiles.DefaultCommandTest` asserts this against golden TP1 / TP2
argv recorded from the 0.34 base. Image `k8v4-up:final-7bbcff2`. Every native library was built from this tree
(`build/`: `libxe2_kv.so` with S1 FAST, batched requests and cascade, `libk8v4_sdpa.so` + `libdnnl.so.3`,
`libgdn_attn_kernels_xe_2.so` (gdn-heads), `libk8v4_w4a8.so` (oneDNN v3.14.2 recipe), and `libk8v4_rssm.so` with
the SLM verify and deferred commit).

## Included

| port | source branch (tip) | knob(s) |
|---|---|---|
| sliced weight uploads | `b70/sliced-upload` (f236b7f) + 0.34 anchor | `K8V4_UPLOAD_SLICE_MB` |
| PR3 decode kernels + multi-page S1 | PR3 commits, `b70/s1-multipage` (0e0307a), as cherry-picked in `b70/e2e-034` | `XE2_KV_S1_SPLITS` (+ `_PAGES`, `_GRF`) |
| faster multi-row S1, batched requests | `b70/s1-rows` (6e438f8) | `XE2_KV_S1_FAST`, `XE2_KV_BATCH_REQS` |
| draft depth by batch size | `b70/mtp-shared-depth` (6255c1d) | `K8V4_MTP_SHARED_DEPTH` |
| host INT8 embedding, draft skips shared tables | `b70/embed-host` (b929ac4), `b70/draft-skip-shared` (1df5454) | `K8V4_EMBED_HOST`, `K8V4_DRAFT_SKIP_SHARED` |
| head-parallel GDN prefill | `b70/gdn-heads` (8a7b370) | `K8V4_GDN_HEADS` (`compile_gdn.sh`) |
| W4A8 prefill: layers, oneDNN s4 / grouped scales, fused quantizers, add+norm | `b70/w4a8-layers` → `-onednn` → `-fused` (d1b0328), `b70/w4a8-addnorm` (a87c5e2) | `K8V4_W4A8_LAYERS/KERNEL/LIB/AGROUP/FUSED` |
| [K, N] weights, reproducible oneDNN matmuls | `b70/ncontig` (5d83f49) | `K8V4_NCONTIG_RE`, `K8V4_ONEDNN_DETERMINISTIC` |
| draft last row, draft vocab (with the s4-head fix), chunked verify, deep draft | `b70/spec-034` (ef53c1f) | `K8V4_DRAFT_LASTROW`, `K8V4_DRAFT_VOCAB`, `K8V4_VERIFY_CHUNK`, `K8V4_DEEP_DRAFT` |
| GDN state dtype, RecoverSSM, SLM verify + deferred commit (incl. the request-slot splice) | `b70/gdn-state-dtype` (a511c3d), `b70/gdn-recoverssm` (59a2204), `b70/rssm-verify` (7637643) | `K8V4_MAMBA_SSM_DTYPE`, `K8V4_GDN_RECOVERSSM`, `K8V4_RSSM_VERIFY`, `K8V4_RSSM_DEFER_COMMIT` |
| deep drafting for several users | `b70/deep-all` (001380d) | `K8V4_DEEP_ALL` |
| prefix-hit granularity | `b70/prefix-match-unit` (6582ec6) | `K8V4_PREFIX_MATCH_UNIT`, `K8V4_PREFIX_JUNCTIONS` |
| agent / OpenAI-API parity | `b70/agent-parity` (e16cee4) | `K8V4_CHAT_TEMPLATE_KWARGS`, `K8V4_CHAT_TEMPLATE_FIX`, `K8V4_PROMPT_TOKENS_DETAILS`, `K8V4_HTTP_KEEP_ALIVE` |
| fused draft decode graph, activation dtype | `b70/draft-fused-decode` (b18727b), `b70/act-dtype` (31b208f) | `K8V4_DRAFT_FUSED_DECODE`, `K8V4_DTYPE` |
| shared-prefix cascade for the drafter batch | `b70/kv-cascade` (5ebc0ed) | `XE2_KV_CASCADE` |

Not included: gemm-warmup, decode-fp16, draft-attn-window, prefill-kv-reuse, rssm-grouped-commit, executor-backend
(its launch test was dropped in the merge), the `*-ownrssm` archives, and every 0.30-only branch. The conflict
resolutions (all in `launch.sh` / `test_launch_profiles.py`) follow `b70/test-rssm-verify2`, `b70/test-dwin`,
`b70/test-prefix-all034` and `b70/test-parity-addnorm-034`.

New on this branch:
- 9362cfb, RecoverSSM SLM verify / deferred commit for windows of up to 16 tokens. `K8V4_RSSM_VERIFY=slm` and
  `K8V4_RSSM_DEFER_COMMIT=1` refused windows over 8 tokens, so they could not start together with deep drafting
  (depth 15 = 16-token windows). The kernel already loops over the window; only its SLM layout is sized by
  `max_t`. Replay buffers of up to 8 positions launch the `max_t = 8` instantiation exactly as before. Buffers of
  9-16 launch `max_t = 16` (29 KB of SLM with the deferred commit). `rssm_defer.MAX_WINDOW` goes from 8 to 16. The
  XPU checks gain 16-token cases.
- 7bbcff2, the golden default-command test.

Tests: the full CPU suite in the vLLM 0.34 image passes (232 tests, 17 skipped without an XPU). The XPU checks ran
on the B70 at the start of the final window (`/root/rice/results/fin-xpu.txt`), and all 9 pass:
`test_rssm_defer` and `test_cascade_xpu`. In `test_rssm_defer`, `verify_slm` is bit-identical to the stock verify,
and the deferred commit is bit-identical to commit-then-verify (outputs, states and conv blocks; fp16 / bf16 /
fp32 state; plain and align mode). Both checks now include 16-token windows (`max_t = 16`). In
`test_cascade_xpu`, an unshared batch is bit-identical, and a shared prefix matches the batched path.

## Config F (launch-ready)

```
K8V4_MAX_SEQS=4 K8V4_PREFIX_CACHING=1 K8V4_MAX_MODEL_LEN=131072 K8V4_GPU_MEMORY_UTILIZATION=0.96
XE2_KV_S1_SPLITS=32 XE2_KV_S1_FAST=1 XE2_KV_BATCH_REQS=1 XE2_KV_CASCADE=1
K8V4_EMBED_HOST=1 K8V4_DRAFT_SKIP_SHARED=1 K8V4_GDN_HEADS=1
K8V4_W4A8_LAYERS=all K8V4_W4A8_KERNEL=onednn K8V4_W4A8_LIB=$PWD/build/libk8v4_w4a8.so
K8V4_W4A8_AGROUP=64 K8V4_W4A8_FUSED=silu,gnorm,addnorm K8V4_ONEDNN_DETERMINISTIC=1
K8V4_DRAFT_LASTROW=1 K8V4_DRAFT_VOCAB=<64K id list, draft_vocab_64k.json> K8V4_MTP_SHARED_DEPTH=3
K8V4_DEEP_DRAFT=12,16 K8V4_DEEP_ALL=1
K8V4_MAMBA_SSM_DTYPE=float16 K8V4_GDN_RECOVERSSM=1 K8V4_RSSM_VERIFY=slm K8V4_RSSM_DEFER_COMMIT=1
K8V4_DRAFT_FUSED_DECODE=1 K8V4_DTYPE=float16 K8V4_PREFIX_MATCH_UNIT=64
K8V4_CHAT_TEMPLATE_KWARGS='{"reasoning_effort":"medium"}' K8V4_CHAT_TEMPLATE_FIX=1
K8V4_PROMPT_TOKENS_DETAILS=1 K8V4_HTTP_KEEP_ALIVE=300
```

Plus `-e K8V4_UPLOAD_SLICE_MB=128` before the image name; without it the model load hangs on this host. The b70
harness inserts it, and the exact command is `/root/k8v4work/fin-F-g64.cmd`. `launch.sh` adds
`K8V4_VERIFY_CHUNK=8` with deep drafting. The speculative config is depth 15 for a lone request or for all-copying
batches, else 3. The capture sizes are `[7,8,12,16,32,48,64]`.

### W4A8 activation scaling: grouped (g64), by QualityBench

`K8V4_W4A8_AGROUP=64` (one int8 scale per token and per 64 input channels), with the fused SiLU / gated-norm /
add-norm quantizers that grouped scales allow. The per-token default was not chosen, based on the QualityBench run
(another agent, `queue2.sh`, results `/root/rice/results/qual-*.txt`):
- lpcheck against the bf16-KV reference: per-token scales drift about 20 times more than g64. The median KL is .12
  against .0066, and top-1 agreement is 30/40 against 37/40.
- QualityBench task scores were within noise for all arms.

Our prod runs the same choice (`B70_W4A8_AGROUP=64` + SILU / GNORM / ADDNORM).

Numerics of F: fp16 activations and fp16 GDN state, as our prod uses, and int8 prefill activations with g64
scales. Everything else is bit-identical or lossless (spec decode).

## Scorecard (one gpu.lock hold, 2026-11-20 13:45-15:30 UTC, arms back to back)

Arms:
- **F**: their stack, config F, image `k8v4-up:final-7bbcff2`, fresh compile cache.
- **OURS-SWIFT**: our 0.34 port serving Swift 1.5, `/root/k8v4work/swift034.cmd`. Same model as F, so this is the
  like-for-like comparison.
- **OURS**: prod -27 command, `exp2.sh` without FULLCMD. It runs its own model (qwen38, Qwen3.8 int4 G64), so
  quality-sensitive rows (EXACT, acceptance-driven decode) are not like for like.

Every arm was run with the same harness: `/root/rice/exp2.sh`, `TESTS="exact needle users prefix custom mt"`,
`USERS="1 4"`, `MT_LEVELS="1 4"`, `CONV_PER_CLIENT=1`, `SEED_BASE=20261118`, `NEEDLE="32000 120000"`. CUSTOM is
`bench/decode_step_timer.py` (warm 63K prefixes, 3 rounds x 512 tokens, 4 then 1 users) plus the coding client
`bench/serve_coding.py` (2K / 8K / 32K / 128K, 3 repeats, 448 tokens). Results are in
`/root/rice/results/fin-{F-g64,swift,ours}.txt` (+ `-coding.jsonl`, `exp-*/serve.log`), and the table was produced
by `/root/k8v4work/final-scorecard.py`.

| metric | F (theirs, config F) | OURS-SWIFT | OURS (prod -27) | F vs OURS | F vs OURS-SWIFT |
|---|---:|---:|---:|---|---|
| KV tokens | **496,630** | 373,124 | 339,639 | ahead (+46%) | ahead (+33%) |
| 63K cold prefill, 1 user (tok/s) | 2,005 | 2,042 | 2,050 | behind (-2.2%) | tie (-1.8%) |
| step p50, 4 users, 63K (ms) | 56.2 | 54.7 | 54.2 | behind (+3.7%) | behind (+2.7%) |
| step p50, 1 user, 63K (ms) | **42.8** | 45.1 | 43.0 | tie | ahead (-5%) |
| 1-user decode, 63K (tok/s) | 77.6 | 71.7 | 93.1 | behind (-17%) | ahead (+8%) |
| 4-user aggregate, 63K (tok/s) | 193.7 | 203.9 | 203.3 | behind (-5%) | behind (-5%) |
| warm TTFT, 1 / 4 users (s) | 0.47 / 2.02 | 0.40 / 1.15 | 0.40 / 1.14 | behind | behind |
| agent sessions, 1 / 4 clients (s) | 158 / 290 | 153 / 282 | 153 / 280 | behind (+3% / +4%) | behind (+3% / +3%) |
| coding decode 2K / 8K / 32K / 128K (tok/s) | 146.7 / 142.9 / 133.4 / 106.7 | 131.0 / 138.1 / 125.2 / 99.2 | 145.1 / 142.1 / 135.4 / 99.8 | tie / tie / tie / ahead (+7%) | ahead at every length (+3..12%) |
| EXACT | 8/9 | 8/9 | 9/9 | behind (other model) | tie |
| NEEDLE 32K + 120K | 6/6 | 6/6 | 6/6 | tie | tie |
| preemptions / suspicious log lines | 0 / 0 | 0 / 0 | 0 / 0 | | |

Tie = within ±2%. Single runs. Earlier windows put run-to-run noise at about ±4% for 4-user aggregates and ±0.1 ms
for step p50.

Prefix check (`prefix_check.py`, cache hit vs cold, 10 prompts):

| | full hit KL mean / top-1 | partial hit KL mean / top-1 |
|---|---|---|
| F | 0.00000 / 10/10, 99% cached | 0.0127 / 10/10, 61% cached |
| OURS-SWIFT | 0.00020 / 10/10 | 0.0083 / 9/10 |
| OURS | 0.00000 / 10/10 | 0.0008 / 10/10 |

## Reading it

- F's clear wins:
  - KV pool: +33-46%, from RecoverSSM + fp16 GDN state + K8/V4 KV.
  - One-user step time and one-user decode against our Swift port: 42.8 vs 45.1 ms, 77.6 vs 71.7 tok/s.
  - Coding-client decode at every length against our Swift port, and at 128K against prod.
  - Prefill and needle are level.
- OURS (prod) leads one-user decode at 63K by 17% (93.1 vs 77.6 tok/s) at the same step time (43.0 vs 42.8 ms).
  That gap is acceptance per step, not kernel time: prod serves its own model with the GPTQ draft bake.
- Four users: F's step is 56.2 ms against 54.1-54.3 for the l3h / l3best configs of the last-3ms and rssm-verify
  docs, and 54.2 / 54.7 for ours. F differs from those configs by deep drafting (depth-15 speculative config, so a
  different page size), the cascade, g64 with fused quantizers, and the prefix match unit. This window does not
  separate them; a follow-up A/B (F minus deep drafting) would. F's 4-user aggregate, 193.7, is 5% below both of ours;
  agent sessions are 3-4% slower.
- Warm TTFT at 4 users (2.02 s vs 1.15 s) is F's largest relative gap. With `K8V4_PREFIX_MATCH_UNIT=64`, one user
  is close (0.47 vs 0.40 s). Not investigated.
- F trades its four-user step for deep copy drafting. The copy workloads are not in this scorecard (see
  docs/2026-11-19-deep-all.md: all-copying 4 x 8K +93%, 4 x 63K +58% over the same base).

## Files

- Window `/root/k8v4work/win-final.sh` (log `/root/k8v4work/win-final.log`), build `/root/k8v4work/bld-final.sh`
  (log `bld-final.log`), table `/root/k8v4work/final-scorecard.py`.
- Commands `/root/k8v4work/fin-F-g64.cmd`, `/root/k8v4work/swift034.cmd`, and
  `/root/rice/results/exp-fin-ours/cmd.txt`.
