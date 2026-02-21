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

- **Batch Processing**: When implementing batch processing functionality:

  - Centralize memory allocation logic in helper methods
  - All memory for C batch structures and their token arrays must be allocated using the C allocator (`LibC.malloc`) to ensure compatibility with `llama_batch_free`.
  - Never mix Crystal's `Pointer.malloc` and C's `malloc` for the same resource.
  - Always release batch memory using `llama_batch_free` (never manually free token arrays from Crystal).
  - Clearly document the ownership of memory resources and ensure that only one owner is responsible for freeing each resource.
  - The `Batch` class should use an `owned` flag to indicate whether it is responsible for freeing the underlying C resource.
  - The `finalize` method must call `llama_batch_free` if and only if `owned` is true.
  - Consider providing simplified high-level APIs for common use cases

- **Circular References**: When objects reference each other (e.g., `Context` and `KvCache`):
  - Implement proper cleanup logic in private methods and call them from `finalize`
  - Consider using weak references where appropriate
  - Document the relationship between objects

## Error Handling for C API Calls

- Include error codes and specific details in exception messages
- For critical operations (model loading, context creation), provide more detailed error information
- When wrapping C functions that return error codes, propagate meaningful error messages

## llama.cpp Version Compatibility

### Version Mapping Rules

- `shard.yml` version must use the numeric build value (example: `8119`).
- Release tags must use `v<build>` format (example: `v8119`).
- When referenced in documentation or scripts, the build is prefixed with `b` (example: `b8119`).

### Version Update Process

Document which version of llama.cpp the library is compatible with. When updating to support a new llama.cpp version:

1. Update `version` in `shard.yml` to the matching numeric build
2. Create/update release tag as `v<build>`
3. Run `assets/download_headers.sh` to download the new header files
4. Update `src/llama/lib_llama.cr` bindings (struct/enum/function signatures)
5. Update wrapper code under `src/llama/` when API behavior changes (especially LoRA-related paths)
6. Ensure workflows are aligned with the current release artifacts (`.tar.gz`) and test asset requirements
7. Verify docs (`README.md`) still match the build/runtime model
8. Run tests:
  - `crystal spec`
  - LoRA specs with adapter path configured when applicable
  - If model loading reports "No backends loaded", set `GGML_BACKEND_PATH` to a backend library file (for example `libggml-cpu-haswell.so`), not a directory
  - Typical local command:
    - `MODEL_PATH=/path/to/model.gguf ADAPTER_PATH=/path/to/adapter.gguf LIBRARY_PATH=/path/to/libs LD_LIBRARY_PATH=/path/to/libs GGML_BACKEND_PATH=/path/to/libs/libggml-cpu-haswell.so crystal spec`
9. Validate examples:
  - `examples/simple.cr`
  - `examples/chat.cr`
  - `examples/tokenize.cr`
10. Commit changes and create a pull request

### Standard Linker/Runtime Environment

- Do not use project-specific linker environment variables.
- Use standard environment variables:
  - Compile/link: `LIBRARY_PATH`
  - Runtime (Linux): `LD_LIBRARY_PATH`
  - Runtime (macOS): `DYLD_LIBRARY_PATH`
- When building against local libraries, prefer explicit flags:
  - `crystal build ... --link-flags "-L<libdir> -Wl,-rpath,<libdir> -lllama -lggml"`
