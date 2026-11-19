# Host-resident INT8 embedding table (`K8V4_EMBED_HOST`, 2026-11-19)

One opt-in change in `k8v4_v030/embed_int8_runtime.py`. With the variable unset or `0` the loader
runs the previous code unchanged (device-resident table).

## Why

The Swift bake's INT8 embedding side table is 248,320 x 5,120 int8 = 1.185 GiB and stays on the GPU.
A step only reads the rows of its own tokens (5 KiB per token), so the table is almost never touched
while it holds memory the KV cache could use. On one B70 at `--gpu-memory-utilization 0.96` the KV
pool is 12.74 GiB = 361,577 tokens (about 28,400 tokens per GiB), so the table is worth about
**+33,600 KV tokens (+9%)** if it leaves the device.

## Change

`K8V4_EMBED_HOST=1`:

- `materialize()` builds the INT8 table directly in pinned (page-locked) host memory instead of a
  pageable CPU tensor.
- The GPU gets a zero-copy view of it, `vllm.utils.torch_utils.get_accelerator_view_from_cpu_tensor`
  (UVA; on XPU `torch.ops._C.get_xpu_view_from_cpu_tensor`). The layer keeps a reference to the
  pinned tensor because the view does not own the pages.
- The per-row scales (0.5 MB) stay on the device. `embedding()` is unchanged: the same
  `F.embedding` gathers, now reading the rows over PCIe.
- The MTP draft shares the target's `embed_tokens` after load (`maybe_share_target_embed`), so it
  reads the same host table.
- CPU layers ignore the flag.

The values gathered are **bit-identical**: the same int8 codes and scales. `launch.sh` passes
`K8V4_EMBED_HOST` only when it is set, and rejects values other than 0/1.

Our own stack has run the same idea since our -3 (BF16 table, `B70_EMBED_UVA`): it gave
about +2.5 GB of KV with no measurable decode or prefill change. vLLM 0.34's
`--cpu-offload-params` does not cover this case. On stock `qwen3_5.py` the UVA offloader only
wraps the decoder layers, and the INT8 loader replaces `layer.weight` after loading anyway, so the
view has to be made here.

Side effect: with `K8V4_DRAFT_SKIP_SHARED` unset, the draft's own INT8 table is also built pinned and
then dropped when the target's table is shared. That is 1.27 GB of host RAM kept by the pinned-host
caching allocator. Use it together with `K8V4_DRAFT_SKIP_SHARED=1`, which makes the draft's table a
one-row placeholder.

## Tests

- `k8v4_v030/tests/test_embed_host.py` (CPU): strict flag parsing; the default path never builds a
  host table; host mode keeps the codes on the host and exactly one view on the device; it is idempotent.
- XPU only: the view gathers the same rows as a device copy, including the padding rows.
- `test_launch_profiles.py`: the default argv is unchanged, and bad values exit with code 2.
- Passes on the 0.30 image and on stock `vllm/vllm-openai-xpu:v0.34.0`.

## Measurements

Not measured yet. The GPU window queued on the 0.30 stack was cancelled when the port target moved to
vLLM 0.34. The plan, to run on the 0.34 base once it serves, is the script
`win-memload.sh` (in the b70 work dir, not in this repo). It runs util 0.96, `K8V4_MAX_SEQS=4`,
prefix caching on, `XE2_KV_S1_SPLITS=32`, and these steps:

1. Read "KV cache size: N tokens" with the knob unset and set. Expected: about +33K tokens.
2. Run the exact battery and needle at 32K/120K. Expected: same results.
3. Run `conc64.py 1 512 63000` and `conc64.py 4 512 63000`. Check that one-user decode and cold
   63K prefill are within noise. The embedding gather is about 21 MB over PCIe for a full
   4,224-token prefill chunk and 140 KB for a 4-user decode step.
