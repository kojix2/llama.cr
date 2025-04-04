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
  - Clearly document the ownership of memory resources
  - Consider providing simplified high-level APIs for common use cases

- **Circular References**: When objects reference each other (e.g., `Context` and `KvCache`):
  - Implement proper cleanup in `finalize` methods
  - Consider using weak references where appropriate
  - Document the relationship between objects

## Error Handling for C API Calls

- Include error codes and specific details in exception messages
- For critical operations (model loading, context creation), provide more detailed error information
- When wrapping C functions that return error codes, propagate meaningful error messages

## llama.cpp Version Compatibility

- Document which version of llama.cpp the library is compatible with
- When updating to support a new llama.cpp version:
  - Update the lib_llama.cr bindings first
  - Test thoroughly with the new version
  - Document any breaking changes
