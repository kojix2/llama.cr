# Example of using advanced sampling methods with llama.cr
#
# This example demonstrates how to use the new sampling methods
# added to llama.cr, including:
# - Extended Temperature sampling
# - Top-N Sigma sampling
# - XTC sampling
# - Infill sampling
# - Grammar Lazy Patterns sampling
#
# Usage:
#   crystal run examples/advanced_sampling_methods.cr -- --model=/path/to/model.gguf

require "../src/llama"

# Get model path from command line arguments
model_path = ARGV.find { |arg| arg.starts_with?("--model=") }.try &.split("=")[1]?

if model_path.nil?
  puts "Please provide a model path with --model=/path/to/model.gguf"
  exit 1
end

# Load the model
puts "Loading model from #{model_path}..."
model = Llama::Model.new(model_path)

# Create a context
puts "Creating context..."
context = model.context

# Define a prompt
prompt = "Write a short story about a robot who learns to feel emotions:"
puts "Prompt: #{prompt}"

# Function to generate text with a specific sampler chain
def generate_with_sampler(context, prompt, sampler_chain, max_tokens = 100)
  result = context.generate_with_sampler(prompt, sampler_chain, max_tokens)
  result
end

# 1. Extended Temperature Sampling
puts "\n=== Extended Temperature Sampling ==="
puts "This sampler provides more control over temperature with dynamic adjustment"

temp_chain = Llama::SamplerChain.new
temp_chain.add(Llama::TopKSampler.new(40))
temp_chain.add(Llama::TempExtSampler.new(0.8_f32, 0.5_f32, 1.0_f32))
temp_chain.add(Llama::DistSampler.new(42))

temp_result = generate_with_sampler(context, prompt, temp_chain, 50)
puts "Result: #{temp_result}"

# 2. Top-N Sigma Sampling
puts "\n=== Top-N Sigma Sampling ==="
puts "This sampler selects tokens based on their distance from the mean in standard deviations"

sigma_chain = Llama::SamplerChain.new
sigma_chain.add(Llama::TopNSigmaSampler.new(2.0_f32))
sigma_chain.add(Llama::TempSampler.new(0.8_f32))
sigma_chain.add(Llama::DistSampler.new(42))

sigma_result = generate_with_sampler(context, prompt, sigma_chain, 50)
puts "Result: #{sigma_result}"

# 3. XTC Sampling
puts "\n=== XTC Sampling ==="
puts "This sampler combines aspects of several sampling methods"

xtc_chain = Llama::SamplerChain.new
xtc_chain.add(Llama::XtcSampler.new(0.3_f32, 0.8_f32, 1, 42))
xtc_chain.add(Llama::DistSampler.new(42))

xtc_result = generate_with_sampler(context, prompt, xtc_chain, 50)
puts "Result: #{xtc_result}"

# 4. Infill Sampling
puts "\n=== Infill Sampling ==="
puts "This sampler is designed for fill-in-the-middle tasks"

# For infill, we need a different prompt structure
infill_prompt = "The robot <FILL> with emotions."
puts "Infill Prompt: #{infill_prompt}"

infill_chain = Llama::SamplerChain.new
infill_chain.add(Llama::TopKSampler.new(40))
infill_chain.add(Llama::TopPSampler.new(0.95_f32, 1))
infill_chain.add(Llama::InfillSampler.new(model.vocab))
infill_chain.add(Llama::DistSampler.new(42))

infill_result = generate_with_sampler(context, infill_prompt, infill_chain, 20)
puts "Result: #{infill_prompt.gsub("<FILL>", infill_result)}"

# 5. Grammar Lazy Patterns Sampling
puts "\n=== Grammar Lazy Patterns Sampling ==="
puts "This sampler applies grammar constraints when triggered by specific patterns"

# Define a simple JSON grammar
json_grammar = %q{
  root ::= object
  object ::= "{" ws (string ":" ws value ("," ws string ":" ws value)*)? "}" ws
  array ::= "[" ws (value ("," ws value)*)? "]" ws
  value ::= object | array | string | number | "true" | "false" | "null"
  string ::= "\"" ([^"\\] | "\\" .)* "\""
  number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?
  ws ::= [ \t\n]*
}

# This prompt will trigger the grammar when "JSON:" appears
grammar_prompt = "Generate information about the robot in JSON format. JSON:"
puts "Grammar Prompt: #{grammar_prompt}"

grammar_chain = Llama::SamplerChain.new
grammar_chain.add(Llama::TopKSampler.new(40))
grammar_chain.add(Llama::GrammarLazyPatternsSampler.new(
  model.vocab, json_grammar, "root", ["JSON:"]
))
grammar_chain.add(Llama::DistSampler.new(42))

grammar_result = generate_with_sampler(context, grammar_prompt, grammar_chain, 100)
puts "Result: #{grammar_result}"

puts "\nAll sampling methods demonstrated successfully!"
