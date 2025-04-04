# Example of using encoder-decoder models with llama.cr
#
# This example demonstrates how to use encoder-decoder models
# such as T5, BART, or other sequence-to-sequence models.
#
# Usage:
#   crystal run examples/encoder_decoder.cr -- --model=/path/to/model.gguf

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

# Check if the model has an encoder
unless model.has_encoder?
  puts "This model does not have an encoder component."
  puts "This example requires an encoder-decoder model like T5 or BART."
  exit 1
end

# Check if the model has a decoder
unless model.has_decoder?
  puts "This model does not have a decoder component."
  puts "This example requires an encoder-decoder model like T5 or BART."
  exit 1
end

# Create a context
puts "Creating context..."
context = model.context

# Get the decoder start token
decoder_start_token = model.decoder_start_token
puts "Decoder start token: #{decoder_start_token}"

# Example input for translation
input_text = "Translate to French: Hello, how are you today?"
puts "Input: #{input_text}"

# Tokenize the input
input_tokens = model.vocab.tokenize(input_text)
puts "Tokenized to #{input_tokens.size} tokens"

# Create a batch for encoding
batch = Llama::Batch.new(input_tokens.size)
batch.add_tokens(input_tokens)

# Encode the input
puts "Encoding input..."
begin
  context.encode(batch)
  puts "Encoding successful"
rescue ex : Llama::Error
  puts "Error during encoding: #{ex.message}"
  exit 1
end

# Create a batch for decoding with the decoder start token
decoder_batch = Llama::Batch.new(1)
decoder_batch.add_token(decoder_start_token, 0)

# Start decoding
puts "Decoding..."
result = ""
max_tokens = 100

max_tokens.times do |i|
  # Process the batch
  context.decode(decoder_batch)

  # Sample the next token
  sampler = Llama::SamplerChain.new
  sampler.add(Llama::TopKSampler.new(40))
  sampler.add(Llama::TopPSampler.new(0.95_f32, 1))
  sampler.add(Llama::TempSampler.new(0.8_f32))
  sampler.add(Llama::DistSampler.new)

  token = sampler.sample(context)
  sampler.accept(token)

  # Check for end of sequence
  break if token == model.vocab.eos

  # Convert token to text and add to result
  token_text = model.vocab.token_to_text(token)
  result += token_text

  # Update batch for next token
  decoder_batch = Llama::Batch.new(1)
  decoder_batch.add_token(token, i + 1)
end

puts "Output: #{result}"
