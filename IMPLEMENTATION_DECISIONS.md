# Implementation decisions for the b10809 high-level API

This document records deliberate departures from
`llama.cr-implementation-plan.md`. It distinguishes correctness guarantees from
future performance and CI work.

## Resource ownership uses weak parent registries

Contexts and adapters keep a strong reference to their model. The model tracks
live children through `WeakRef` entries and closes every still-live child during
explicit `Model#close`.

A strong model-to-child registry was tested first. Combined with the required
child-to-model reference and finalizers, it creates a finalizable cycle. Boehm
GC warns about that cycle and may skip its finalizers. The weak registry retains
deterministic explicit close behavior without making leaked wrapper cycles
uncollectable.

## Session and Chat favor canonical text over incremental KV reuse

`Session` is stateful at the public API: each call continues from committed
text, `reset` makes it independent, and snapshots restore that transcript.
Before generation it rebuilds native state from canonical text using checked
decode instead of retaining an incremental token/KV ledger.

This is deliberate. A text stop can end inside one token's byte piece. Keeping
incremental KV state would require removing every affected token, retokenizing
the visible suffix, and transactionally decoding replacements. Rebuilding is
slower but prevents hidden stop bytes from contaminating a later turn. Chat also
renders complete candidate history and commits it only after success.

Snapshots therefore store canonical text, the reported llama.cpp version, and
a model metadata fingerprint. They do not claim native KV portability.
`OverflowPolicy::Shift` is rejected explicitly; silently treating it as the
default error policy would be misleading.

## Embedding batching has explicit native limits

`Embedder#embed_all` uses llama.cpp multi-sequence batches and copies every
vector before the next native call. The b10809 runtime was measured with the
test GGUF: single and batched output matched exactly, output dimension was 288,
and normalized norm was 0.99999999.

Sentence input larger than `n_batch` is rejected. In this build, splitting such
an input changes mean/CLS/last pooling semantics. `Pooling::None` is also
rejected by `Embedder` because it produces token-level vectors rather than one
sentence vector; the advanced `Context#get_embeddings_ith` API remains for that
case.

## Templates are recognized shapes, not arbitrary Jinja

`llama_chat_apply_template` in b10809 recognizes a predefined set of template
shapes. It is not a general Jinja evaluator. Unsupported custom templates raise
`TemplateError`. Tests and the fallback example use a recognized ChatML shape.

The previous wrapper also freed Crystal GC memory with `LibC.free` after a
successful template application. The high-level work exposed the resulting
process abort; buffers now remain under Crystal ownership.

## Runtime checks cannot prove the exact build

The linked library reports semantic version `0.4.0`, not build `b10809`.
`check_compatibility!` rejects another reported stable version before wrappers
pass ABI-sensitive structs by value. Exact build enforcement remains in the
package pin and the C-versus-Crystal layout CI probe.

## No general C adapter or callback queue

Model loading and decode with two worker threads were instrumented on the pinned
runtime. Every observed log callback ran on the calling thread. Callback
exceptions are now caught at the C boundary and retrievable from Crystal.

That evidence does not justify a general native adapter or C-owned log queue.
The callback remains documented as experimental; a queue becomes warranted if
a supported runtime invokes it from an unmanaged worker thread.

## Deferred infrastructure

- A generator-wide injectable fake backend was not added. It would duplicate a
  broad statically bound FFI surface solely for tests. Pure policy components,
  decode-code classification, real-model integration tests, and the ABI probe
  cover the current implementation without introducing a second backend API.
- A Linux ASan job requires building and loading a sanitized b10809 llama.cpp,
  including a correctly ordered sanitizer runtime. This environment uses a
  packaged unsanitized library, so an unexercised workflow would provide a false
  guarantee. `GC_DEBUG=1`, macOS/Linux native jobs, and ABI checks remain; the
  ASan job should be added only with a reproducible sanitized build script.
- Incremental common-prefix KV reuse remains a performance optimization. It is
  gated on token-level stop reconciliation tests and does not weaken current
  transcript correctness.
