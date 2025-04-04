# Development Guidelines

This document outlines the development guidelines for the llama.cr project, primarily intended for AI assistants but also useful for human contributors.

## Language Requirements

- All code, comments, documentation, and commit messages must be written in English
- This includes inline comments, method documentation, and README files
- The project is intended for international use, so avoid language-specific idioms

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

## C Bindings Guidelines

- Use proper Crystal types that correspond to C types
- Use `Pointer(T)` for C pointers
- Use `LibC::SizeT` for `size_t`
- Handle null pointers appropriately
- Ensure proper memory management for allocated resources
