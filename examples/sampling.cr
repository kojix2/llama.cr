#!/usr/bin/env crystal

# Sampling example using llama.cr
#
# Compilation:
#   crystal build examples/sampling.cr --link-flags="-L/path/to/llama.cpp/build/bin"
#
# Execution:
#   LD_LIBRARY_PATH=/path/to/llama.cpp/build/bin ./sampling /path/to/model.gguf "Your prompt here"

require "../src/llama"

# Check command line arguments
if ARGV.size < 2
  puts "Sampling Example for llama.cr"
  puts
  puts "Compilation:"
  puts "  crystal build examples/sampling.cr --link-flags=\"-L/path/to/llama.cpp/build/bin\""
  puts
  puts "Execution:"
  puts "  LD_LIBRARY_PATH=/path/to/llama.cpp/build/bin ./sampling /path/to/model.gguf \"Your prompt here\""
  puts
  puts "Parameters:"
  puts "  1. Path to model file (required)"
  puts "  2. Text prompt (required)"
  puts "  3. Maximum number of tokens to generate (optional, default: 128)"
  exit 1
end

model_path = ARGV[0]
prompt = ARGV[1]
max_tokens = (ARGV[2]? || "128").to_i

puts "Loading model from #{model_path}..."
model = Llama::Model.new(model_path)
puts "Model loaded successfully!"

puts "\nCreating context..."
context = model.context
puts "Context created successfully!"

puts "\nCreating sampler chain..."
sampler = Llama::SamplerChain.new

# Add samplers to the chain
sampler.add(Llama::TopKSampler.new(40))
sampler.add(Llama::TopPSampler.new(0.95, 1))
sampler.add(Llama::TempSampler.new(0.8))
sampler.add(Llama::DistSampler.new)

puts "\nGenerating text from prompt: '#{prompt}'"
puts "Parameters:"
puts "  - Max tokens: #{max_tokens}"
puts "  - Sampling: Top-K(40) -> Top-P(0.95) -> Temp(0.8) -> Dist"

start_time = Time.monotonic
response = context.generate_with_sampler(prompt, sampler, max_tokens)
end_time = Time.monotonic

puts "\nGenerated text:"
puts "#{prompt}#{response}"
puts "\nGeneration took #{(end_time - start_time).total_seconds.round(2)} seconds"
