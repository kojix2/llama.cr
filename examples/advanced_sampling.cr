#!/usr/bin/env crystal

# Advanced sampling methods example using llama.cr
#
# This example demonstrates how to use the advanced sampling methods:
# - Min-P sampling
# - Typical sampling
# - Mirostat sampling
# - Grammar-based sampling
# - Penalties sampling
#
# Compilation:
#   crystal build examples/advanced_sampling.cr --link-flags="-L/path/to/llama.cpp/build/bin"
#
# Execution:
#   LD_LIBRARY_PATH=/path/to/llama.cpp/build/bin ./advanced_sampling /path/to/model.gguf

require "../src/llama"

# Check command line arguments
if ARGV.size < 1
  puts "Advanced Sampling Methods Example for llama.cr"
  puts
  puts "Compilation:"
  puts "  crystal build examples/advanced_sampling.cr --link-flags=\"-L/path/to/llama.cpp/build/bin\""
  puts
  puts "Execution:"
  puts "  LD_LIBRARY_PATH=/path/to/llama.cpp/build/bin ./advanced_sampling /path/to/model.gguf"
  puts
  puts "Parameters:"
  puts "  1. Path to model file (required)"
  exit 1
end

model_path = ARGV[0]

# Sample prompts for testing different sampling methods
prompts = {
  "creative" => "Write a short poem about artificial intelligence:",
  "factual"  => "Explain the concept of quantum computing in simple terms:",
  "coding"   => "Write a function to calculate the Fibonacci sequence in Python:",
  "grammar"  => "List the top 5 programming languages in 2023:",
}

puts "Loading model from #{model_path}..."
model = Llama::Model.new(model_path)
puts "Model loaded successfully!"
puts "  - Parameters: #{model.n_params}"
puts "  - Embedding size: #{model.n_embd}"

# Create a context
puts "\nCreating context..."
params = Llama::Context.default_params
context = model.context(params)

# ===== Min-P Sampling =====
puts "\n===== Min-P Sampling ====="
puts "Min-P sampling keeps tokens with probability >= p * max_probability"
puts "This can produce more diverse outputs than top-p while avoiding low-probability tokens"

prompt = prompts["creative"]
puts "\nPrompt: #{prompt}"

# Create a sampler chain with Min-P
chain = Llama::SamplerChain.new
chain.add(Llama::TopKSampler.new(40))      # First filter with top-k
chain.add(Llama::MinPSampler.new(0.05, 1)) # Then apply min-p with p=0.05
chain.add(Llama::TempSampler.new(0.8))     # Apply temperature
chain.add(Llama::DistSampler.new(42))      # Final distribution sampling

# Generate text
puts "\nGenerating with Min-P sampling..."
result = context.generate_with_sampler(prompt, chain, 100)
puts "\nResult:\n#{result}"

# ===== Typical Sampling =====
puts "\n===== Typical Sampling ====="
puts "Typical sampling selects tokens based on their 'typicality' (entropy-based)"
puts "It can produce more natural and less repetitive text"

prompt = prompts["factual"]
puts "\nPrompt: #{prompt}"

# Create a sampler chain with Typical sampling
chain = Llama::SamplerChain.new
chain.add(Llama::TopKSampler.new(40))         # First filter with top-k
chain.add(Llama::TypicalSampler.new(0.95, 1)) # Then apply typical with p=0.95
chain.add(Llama::TempSampler.new(0.7))        # Apply temperature
chain.add(Llama::DistSampler.new(42))         # Final distribution sampling

# Generate text
puts "\nGenerating with Typical sampling..."
result = context.generate_with_sampler(prompt, chain, 100)
puts "\nResult:\n#{result}"

# ===== Mirostat Sampling =====
puts "\n===== Mirostat Sampling ====="
puts "Mirostat dynamically adjusts the sampling temperature to maintain a target entropy"
puts "It can produce more consistent quality outputs"

prompt = prompts["creative"]
puts "\nPrompt: #{prompt}"

# Create a sampler chain with Mirostat V2
chain = Llama::SamplerChain.new
chain.add(Llama::TopKSampler.new(40))                 # First filter with top-k
chain.add(Llama::MirostatV2Sampler.new(42, 5.0, 0.1)) # Mirostat V2 with tau=5.0, eta=0.1
chain.add(Llama::DistSampler.new(42))                 # Final distribution sampling

# Generate text
puts "\nGenerating with Mirostat V2 sampling..."
result = context.generate_with_sampler(prompt, chain, 100)
puts "\nResult:\n#{result}"

# ===== Penalties Sampling =====
puts "\n===== Penalties Sampling ====="
puts "Penalties sampling applies various penalties to token probabilities"
puts "It can reduce repetition and improve diversity"

prompt = prompts["factual"]
puts "\nPrompt: #{prompt}"

# Create a sampler chain with Penalties
chain = Llama::SamplerChain.new
chain.add(Llama::TopKSampler.new(40))                     # First filter with top-k
chain.add(Llama::TopPSampler.new(0.95, 1))                # Then apply top-p
chain.add(Llama::PenaltiesSampler.new(64, 1.1, 0.0, 0.0)) # Apply penalties: last_n=64, repeat=1.1
chain.add(Llama::TempSampler.new(0.8))                    # Apply temperature
chain.add(Llama::DistSampler.new(42))                     # Final distribution sampling

# Generate text
puts "\nGenerating with Penalties sampling..."
result = context.generate_with_sampler(prompt, chain, 100)
puts "\nResult:\n#{result}"

# ===== Grammar Sampling =====
puts "\n===== Grammar Sampling ====="
puts "Grammar sampling constrains generation to follow a formal grammar"
puts "It's useful for generating structured text like code or lists"

# Simple grammar for a numbered list
grammar = <<-GRAMMAR
root ::= list
list ::= item+
item ::= number ". " sentence "\\n"
number ::= "1" | "2" | "3" | "4" | "5"
sentence ::= [^\\n]+
GRAMMAR

prompt = prompts["grammar"]
puts "\nPrompt: #{prompt}"
puts "Using grammar for a numbered list"

# NOTE: Grammar sampling is currently disabled due to a memory issue
puts "\nGenerating with Grammar sampling... (DISABLED - using Top-P instead)"

# Create a sampler chain with Top-P instead of Grammar
chain = Llama::SamplerChain.new
chain.add(Llama::TopKSampler.new(40))      # First filter with top-k
chain.add(Llama::TopPSampler.new(0.95, 1)) # Then apply top-p
chain.add(Llama::TempSampler.new(0.8))     # Apply temperature
chain.add(Llama::DistSampler.new(42))      # Final distribution sampling

# Generate text
result = context.generate_with_sampler(prompt, chain, 150)
puts "\nResult:\n#{result}"

puts "\nAdvanced sampling methods example completed!"
