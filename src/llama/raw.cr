# Upstream-tracking, ABI-sensitive llama.cpp bindings.
#
# Prefer `require "llama"` and the supported wrappers for application code.
# LibLlama declarations, pointer lifetimes, and value-passed structs may change
# whenever the pinned llama.cpp build changes.
require "./lib_llama"
