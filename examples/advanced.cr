#!/usr/bin/env crystal

# Advanced features example using llama.cr
#
# This example demonstrates the use of advanced features:
# - KV cache management
# - Batch processing
# - State management
#
# Compilation:
#   crystal build examples/advanced.cr --link-flags="-L/path/to/llama.cpp/build/bin"
#
# Execution:
#   LD_LIBRARY_PATH=/path/to/llama.cpp/build/bin ./advanced /path/to/model.gguf

require "../src/llama"
require "../src/llama/chat"

# Check command line arguments
if ARGV.size < 1
  puts "Advanced Features Example for llama.cr"
  puts
  puts "Compilation:"
  puts "  crystal build examples/advanced.cr --link-flags=\"-L/path/to/llama.cpp/build/bin\""
  puts
  puts "Execution:"
  puts "  LD_LIBRARY_PATH=/path/to/llama.cpp/build/bin ./advanced /path/to/model.gguf"
  puts
  puts "Parameters:"
  puts "  1. Path to model file (required)"
  exit 1
end

model_path = ARGV[0]

puts "Loading model from #{model_path}..."
model = Llama::Model.new(model_path)
puts "Model loaded successfully!"
puts "  - Parameters: #{model.n_params}"
puts "  - Embedding size: #{model.n_embd}"
puts "  - Layers: #{model.n_layer}"
puts "  - Attention heads: #{model.n_head}"

puts "\nCreating context..."
context = model.context
puts "Context created successfully!"

# ===== KV Cache Management Example =====
puts "\n===== KV Cache Management ====="

# Get the KV cache
kv_cache = context.kv_cache
puts "Initial KV cache state:"
puts "  - Number of tokens: #{kv_cache.n_tokens}"
puts "  - Used cells: #{kv_cache.used_cells}"

# Process a simple prompt
prompt = "Hello, world!"
tokens = model.vocab.tokenize(prompt)
puts "\nTokenized prompt: #{prompt}"
puts "  - Tokens: #{tokens.inspect}"

# Create a batch for the tokens
batch = Llama::Batch.get_one(tokens)
puts "\nCreated batch with #{batch.n_tokens} tokens"

# Process the batch
puts "Processing batch..."
context.decode(batch)
puts "Batch processed successfully!"

# Check KV cache state after processing
puts "\nKV cache state after processing:"
puts "  - Number of tokens: #{kv_cache.n_tokens}"
puts "  - Used cells: #{kv_cache.used_cells}"

# Clear the KV cache
puts "\nClearing KV cache..."
kv_cache.clear
puts "KV cache cleared!"
puts "  - Number of tokens: #{kv_cache.n_tokens}"
puts "  - Used cells: #{kv_cache.used_cells}"

# ===== Batch Processing Example =====
puts "\n===== Batch Processing ====="

# Tokenize the input
tokens = model.vocab.tokenize("Hello")
puts "Tokenized 'Hello':"
puts "  - Tokens: #{tokens.inspect}"

# Create a custom batch with the tokens
puts "Creating a custom batch..."
custom_batch = Llama::Batch.for_tokens(tokens)

puts "Custom batch created with #{custom_batch.n_tokens} tokens"

# Process the custom batch
puts "Processing custom batch..."
context.decode(custom_batch)
puts "Custom batch processed successfully!"

# ===== State Management Example =====
puts "\n===== State Management ====="

# Get the state manager
state = context.state
puts "Getting state size..."
state_size = state.size
puts "State size: #{state_size} bytes"

# Save state to a file
puts "\nSaving state to file..."
state_file = "llama_state.bin"
state.save_file(state_file, tokens)
puts "State saved to #{state_file}"

# Clear the KV cache again
kv_cache.clear

# Load state from file
puts "\nLoading state from file..."
loaded_tokens = state.load_file(state_file)
puts "State loaded successfully!"
puts "  - Loaded tokens: #{loaded_tokens.inspect}"

# Check KV cache state after loading
puts "\nKV cache state after loading state:"
puts "  - Number of tokens: #{kv_cache.n_tokens}"
puts "  - Used cells: #{kv_cache.used_cells}"

# Clean up
File.delete(state_file) if File.exists?(state_file)
puts "\nCleaned up temporary files"

puts "\nAdvanced features demonstration completed!"
