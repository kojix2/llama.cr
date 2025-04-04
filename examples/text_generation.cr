#!/usr/bin/env crystal

# Text generation example using llama.cr
#
# Compilation:
#   crystal build examples/text_generation.cr --link-flags="-L/path/to/llama.cpp/build/bin"
#
# Execution:
#   LD_LIBRARY_PATH=/path/to/llama.cpp/build/bin ./text_generation /path/to/model.gguf "Your prompt here"
#
# Optional parameters:
#   3rd argument: Maximum number of tokens to generate (default: 128)
#   4th argument: Temperature parameter (default: 0.8)

require "../src/llama"

# Check command line arguments
if ARGV.size < 2
  puts "Text Generation Example for llama.cr"
  puts
  puts "Compilation:"
  puts "  crystal build examples/text_generation.cr --link-flags=\"-L/path/to/llama.cpp/build/bin\""
  puts
  puts "Execution:"
  puts "  LD_LIBRARY_PATH=/path/to/llama.cpp/build/bin ./text_generation /path/to/model.gguf \"Your prompt here\""
  puts
  puts "Parameters:"
  puts "  1. Path to model file (required)"
  puts "  2. Text prompt (required)"
  puts "  3. Maximum number of tokens to generate (optional, default: 128)"
  puts "  4. Temperature (optional, default: 0.8)"
  puts
  puts "Example:"
  puts "  ./text_generation tiny_model.gguf \"Once upon a time\" 50 0.7"
  exit 1
end

model_path = ARGV[0]
prompt = ARGV[1]
max_tokens = (ARGV[2]? || "128").to_i
temperature = (ARGV[3]? || "0.8").to_f32

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

puts "\nGenerating text from prompt: '#{prompt}'"
puts "Parameters:"
puts "  - Max tokens: #{max_tokens}"
puts "  - Temperature: #{temperature}"

start_time = Time.monotonic
response = context.generate(prompt, max_tokens: max_tokens, temperature: temperature)
end_time = Time.monotonic

puts "\nGenerated text:"
puts "#{prompt}#{response}"
puts "\nGeneration took #{(end_time - start_time).total_seconds.round(2)} seconds"
