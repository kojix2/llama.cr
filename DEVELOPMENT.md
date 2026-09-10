# Development Guidelines

This document outlines the development guidelines for the llama.cr project, primarily intended for AI assistants but also useful for human contributors.

## Language Requirements

- IMPORTANT: All code, comments, documentation, and commit messages must be written in English

## Crystal-Specific Guidelines

- Place all `require` statements at the top of the file, before any module or class definitions
- Avoid dynamic requires as they are not supported in Crystal
- Follow Crystal's standard naming conventions:
  - Classes and modules use `PascalCase`
  - Methods and variables use `snake_case`
  - Constants use `SCREAMING_SNAKE_CASE`
- Use proper type annotations for method parameters and return values
- Handle memory management appropriately for C bindings (use `finalize` methods)

## Project Structure

- C bindings go in `src/llama/lib_llama.cr`
- Crystal wrapper classes go in their own files under `src/llama/`
- Tests go in the `spec/` directory

## Documentation

- Document all public methods with clear descriptions of parameters and return values
- Include examples where appropriate
- Keep the README.md updated with installation and usage instructions

## Markdown Style Guidelines

- Do not indent code blocks (code blocks should start at the beginning of the line)
- Blank lines before and after code blocks are acceptable
- Use numbered lists for sequential steps
- Use bullet points for non-sequential items
- Use proper heading levels (# for title, ## for sections, ### for subsections)
- Include language specifiers in code blocks (```crystal, ```bash, etc.)

## C Bindings Guidelines

- Use proper Crystal types that correspond to C types
- Use `Pointer(T)` for C pointers
- Use `LibC::SizeT` for `size_t`
- Handle null pointers appropriately
- Ensure proper memory management for allocated resources

## Memory Management for Complex Objects

- Prefer deterministic, idempotent `close` methods and block APIs. Finalizers
  are a fallback only.
- A native-backed child keeps its parent alive. Parent registries use weak
  references so parent/child finalizers do not form a Boehm GC cycle; explicit
  parent close must still close every live child in dependency order.
- Borrowed wrappers and pointer-backed results must validate their owner and be
  copied before another native call can invalidate them.

- **Batch Processing**: When implementing batch processing functionality:

  - Centralize memory allocation logic in helper methods
  - All memory for C batch structures and their token arrays must be allocated using the C allocator (`LibC.malloc`) to ensure compatibility with `llama_batch_free`.
  - Never mix Crystal's `Pointer.malloc` and C's `malloc` for the same resource.
  - Always release batch memory using `llama_batch_free` (never manually free token arrays from Crystal).
  - Clearly document the ownership of memory resources and ensure that only one owner is responsible for freeing each resource.
  - The `Batch` class should use an `owned` flag to indicate whether it is responsible for freeing the underlying C resource.
  - The `finalize` method must call `llama_batch_free` if and only if `owned` is true.
  - Consider providing simplified high-level APIs for common use cases

- **Circular References**: When objects reference each other (e.g., `Context` and `Memory`):
  - Implement proper cleanup logic in private methods and call them from `finalize`
  - Consider using weak references where appropriate
  - Document the relationship between objects

## Error Handling for C API Calls

- Include error codes and specific details in exception messages
- For critical operations (model loading, context creation), provide more detailed error information
- When wrapping C functions that return error codes, propagate meaningful error messages
- Library-owned generation and embedding paths must use checked `decode!` or
  `encode!`; do not read logits or embeddings after a non-zero native result.

## High-Level API Constraints

- Keep `Llama.generate`, `Context`, `Batch`, `State`, and manual samplers
  available alongside additive typed helpers.
- `Session` and `Chat` currently rebuild native state from canonical visible
  text. Do not add incremental KV reuse until stop sequences ending inside a
  token are reconciled transactionally.
- Session snapshots contain canonical text and compatibility metadata, not
  portable native KV bytes.
- `Embedder` copies vectors before subsequent native calls. Do not split one
  sentence across batches for mean, CLS, last, or rank pooling; that changes
  pooling semantics. Token-level `Pooling::None` remains an advanced `Context`
  use case.
- b10809 chat templates are recognized template shapes, not arbitrary Jinja.
  Unsupported shapes must raise `TemplateError`.
- Custom native log callbacks are experimental. Exceptions must never cross the
  C boundary. Add a C-owned queue only if a supported runtime is shown to invoke
  callbacks from unmanaged worker threads.
- Do not add a general C adapter or a second injectable backend abstraction
  without a concrete ABI, callback-threading, or multi-version requirement.

## llama.cpp Version Compatibility

### Version Mapping Rules

- `shard.yml` version must use `0.<build>.<patch>` format (example: `0.10809.1`).
- Increment `<patch>` for wrapper fixes and additive APIs that retain the same
  llama.cpp build. Reset it to `0` when `<build>` changes.
- Release tags must match the shard version with a `v` prefix (example: `v0.10809.1`).
- When referenced in documentation or scripts, the build is prefixed with `b` (example: `b<build>`).
- llama.cpp also publishes stable semver tags (`vX.Y.Z`) that point to a specific build (example: `v0.4.0` points to `b10809`). Mention the mapping when the targeted build is such a stable release.
- Runtime compatibility checks must remain opt-in. Users may load other
  llama.cpp versions to evaluate their actual compatibility; do not reject them
  during normal initialization solely because their version string differs.

### Version Update Process

Document which version of llama.cpp the library is compatible with. When updating to support a new llama.cpp version:

1. Run `crystal run assets/download_headers.cr -- <build>` to update `shard.yml` and download the matching headers
2. Create/update the release tag as `v0.<build>.<patch>`
3. Review the reported header diff and update `src/llama/lib_llama.cr` bindings (struct/enum/function signatures)
4. Update wrapper code under `src/llama/` when API behavior changes (especially LoRA-related paths)
5. Run `crystal run scripts/check_abi.cr` to verify C and Crystal struct layouts
6. Ensure workflows are aligned with the current release artifacts (`.tar.gz`) and test asset requirements
7. Check `llama_version()` in both the official release archive and supported
   package builds, and record any known version-string variants explicitly
8. Verify docs (`README.md`) still match the build/runtime model
9. Run tests:
  - `crystal spec`
  - LoRA specs with adapter path configured when applicable
  - If model loading reports "No backends loaded", set `GGML_BACKEND_PATH` to a backend library file (for example `libggml-cpu-haswell.so`), not a directory
  - Typical local command:
    - `MODEL_PATH=/path/to/model.gguf ADAPTER_PATH=/path/to/adapter.gguf LIBRARY_PATH=/path/to/libs LD_LIBRARY_PATH=/path/to/libs GGML_BACKEND_PATH=/path/to/libs/libggml-cpu-haswell.so crystal spec`
10. Validate examples:
  - `examples/simple.cr`
  - `examples/minimal.cr`
  - `examples/chat.cr`
  - `examples/streaming.cr`
  - `examples/embedding.cr`
  - `examples/tokenize.cr`
  - `examples/server.cr` (build only; dependencies are in `examples/shard.yml`)
10. Commit changes and create a pull request

### Standard Linker/Runtime Environment

- Do not use project-specific linker environment variables.
- Use standard environment variables:
  - Compile/link: `LIBRARY_PATH`
  - Runtime (Linux): `LD_LIBRARY_PATH`
  - Runtime (macOS): `DYLD_LIBRARY_PATH`
- When building against local libraries, prefer explicit flags:
  - `crystal build ... --link-flags "-L<libdir> -Wl,-rpath,<libdir> -lllama -lggml"`
