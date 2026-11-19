# MTP draft skips the tables it shares with the target (`K8V4_DRAFT_SKIP_SHARED`, 2026-11-19)

One opt-in change, made of splices applied by `k8v4_v030/patch_installed_vllm.py` plus a new module
`k8v4_v030/draft_shared.py`. With the variable unset or `0`, the draft loads exactly as before.

## Why

vLLM (0.30, and stock 0.34 unchanged) builds the Qwen3.5 MTP draft with its own full
`embed_tokens` (248,320 x 5,120) and `lm_head`, and loads both from the checkpoint. Then, in
`load_eagle_model`, it deletes them and points the draft at the target's modules, because
`Qwen3_5MTP` has neither `has_own_embed_tokens` nor `has_own_lm_head`. With the Swift bake the
throwaway work per start is:

- 2.37 GiB BF16 embedding copied to the GPU;
- the 1.185 GiB INT8 side table built and uploaded again by `materialize()`;
- the 0.615 GiB GPTQ int4 `lm_head` uploaded.

Those are the largest single host-to-device copies of the draft load, the pattern behind the
sliced-upload hang. They cost load time, but no steady-state memory: the KV pool is sized after the
draft's copies are freed.

## Change

`K8V4_DRAFT_SKIP_SHARED=1`:

- `qwen3_5_mtp.py`: the draft's `embed_tokens` and `lm_head` are built with one vocabulary row
  (`draft_shared.table_rows`). They stay full size if the embeddings are tied, or under pipeline
  parallelism, where vLLM does not share them.
- `qwen3_5_mtp.py` `load_weights`: `draft_shared.load_without_shared` drops the checkpoint tensors
  named `embed_tokens` / `lm_head` when the modules are placeholders. It zero-fills the
  placeholders, so `g_idx` and the other buffers hold defined values for
  `process_weights_after_loading`, and reports their names as loaded.
- `v1/worker/gpu/spec_decode/eagle/utils.py` `load_eagle_model`: after sharing,
  `draft_shared.check_replaced` raises if a placeholder survived. A future vLLM that stopped sharing
  would then fail loudly instead of drafting from a one-row table.

Serving is identical: the draft reads the target's tables either way (**bit-identical**). Every
splice is anchored and idempotent, and refuses unknown source without touching the file. The
anchors match vLLM 0.30 and stock `vllm/vllm-openai-xpu:v0.34.0` unchanged (checked by
`test_installed_vllm_sources_take_the_splice` in both images).

## Tests

- `k8v4_v030/tests/test_draft_shared.py`: flag parsing; `table_rows` (tied, pipeline parallel);
  the full tables load everything; the placeholder load skips the shared tensors and counts them
  as loaded; `check_replaced`; the splice patches once, compiles, refuses unknown source and
  skips missing files; it applies to the installed vLLM.
- `test_launch_profiles.py`: the default argv is unchanged, and bad values exit with code 2.

## Measurements

Not measured yet: the 0.30 window was cancelled when the target moved to vLLM 0.34. To measure:

- the draft's "Loading weights took" line, the knob unset vs set;
- "Model loading took N GiB": should be unchanged, because the copies were freed before;
- whether the second `[swift-embed-int8]` line shows the 1-row draft table;
- start reliability over several starts without `K8V4_UPLOAD_SLICE_MB`.
