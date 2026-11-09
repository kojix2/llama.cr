# llama.cpp v0.9.0 migration

The previous target was v0.4.0 / b10809
(`5266f24da75dc449bd56cbed7addb9c8e4a6a73e`). The new target is
v0.9.0 / b12110 (`cd959bae99ff4bed22bef2604f45241c09bbb8c8`). The stable
and build tags point to the same commit, and their `include/llama.h` files
are identical. The shard version is `0.12110.0`.

Sources: [release notes](https://github.com/ggml-org/llama.cpp/releases/tag/v0.9.0),
[complete comparison](https://github.com/ggml-org/llama.cpp/compare/b10809...b12110),
[public header](https://github.com/ggml-org/llama.cpp/blob/v0.9.0/include/llama.h).
The comparison below covers the full migration from v0.4.0, including changes
introduced before the v0.9.0 release itself.

## Low-level API

| Upstream change | Binding update |
| --- | --- |
| `llama_context_params.moe_cache_size` inserted after `type_v` | Add `LibC::SizeT` at the matching offset; all later offsets change |
| `llama_perf_context_data.n_reallocs_same` and `n_reallocs_diff` | Extend the struct returned by value; verify size and offsets |
| Extended batch API | Add opaque `LlamaBatchExt`, `LlamaEmbd`, `LlamaProcessType`, every public `llama_batch_ext_*` function, and `llama_process` |
| `llama_get_causal_attn` | Add the context attention getter |
| `llama_adapter_lora_init_from_file_ptr` | Add the function with an opaque C `FILE` pointer; also expose the existing model FILE loader |
| `LLAMA_VOCAB_TYPE_TEST` / `PLAMO3` | Add enum values 7 / 8 |
| `LLAMA_FTYPE_MOSTLY_F8_E4M3` | Add enum value 42 |
| `LLAMA_DRY_MAX_SEQ_BREAKERS` | Expose the limit of 64 distinct sequence breakers |
| Session / sequence-state formats | Update version constants from 10 / 3 to 11 / 4 |
| `llama_sampler_chain_n` returns `int32_t` | Existing `Int32` declaration already matches |

The header review also found an existing incorrect `Int32` declaration for
`llama_model_tensor_buft_override.buft`; it is now a pointer, matching upstream.
`scripts/check_abi.cr` checks the changed structs as well as the existing model
and quantization parameter structs.

The vendored ggml headers now match the ggml v0.29.0 sources embedded in b12110.
Changes include FP8 E4M3, precision controls, flash-attention row selection,
batched tensor transfers, scheduler callbacks, and multi-process communication.
These ggml APIs are reference headers; llama.cr does not provide general ggml
tensor or backend bindings. Multimodal `mtmd` and server HTTP APIs remain
outside this shard's API.

## High-level API

- `Context` and `ContextOptions` accept `moe_cache_size` in bytes, defaulting to
  zero. Typed completion, session, chat, and embedder construction forward it.
- `Context#causal_attention?` and `#causal_attention=` expose attention state.
  Changing it rejects overlapping high-level operations. Checked encode/decode
  reject non-causal batches larger than `n_ubatch` before upstream can assert.
- `Context#perf` copies the native diagnostic counters, including the new graph
  reallocation fields. Application timing still uses `Generation#usage`.
- Pooled embedding contexts no longer produce logits. `Embedder` already reads
  only embeddings, so it retains its existing batching and copy semantics.
- The experimental extended batch interface is available in `LibLlama`.
  Existing `Batch`, `Session`, `Chat`, and `Embedder` keep the legacy batch path,
  which remains supported upstream. Extended batches borrow their context's
  memory; free them before closing that context. Input embeddings are copied by
  the native setters, and positions and output flags must be set explicitly.
- Recreate old native state files. Portable session snapshots also reject
  mismatched wrapper/native version metadata.
- Compatibility checks remain opt-in and accept exactly `0.9.0` (Homebrew)
  and `0.9.0-dev` (official b12110 archive), both checked locally.

## Packaging and validation

CI derives b12110 from `shard.yml`, verifies the official Linux/macOS archive
SHA-256 digests, and pins the ASan source build to the commit above. The macOS
backend search also recognizes the archive's unsuffixed `libggml-cpu.dylib`.

Regression coverage includes the extended batch token/embedding paths,
attention mode changes and micro-batch limits, graph counter returns, runtime
compatibility, generation, streaming, chat, embeddings, and state round trips.
Disabled MoE cache policy is exercised on the dense test model; typed options
also cover sizes larger than 4 GiB. This does not validate expert offloading on
a real MoE model. LoRA integration requires its separate
base-model and adapter fixtures.

Local validation on macOS arm64 passed with Homebrew v0.9.0 and the official
b12110 archive: 156 regular examples and 11 LoRA examples, including the new C
FILE loader. All seven examples build; the six CLI examples run with the small
model. ABI layouts and Crystal formatting checks pass. Linux and ASan execution
remain CI checks.
