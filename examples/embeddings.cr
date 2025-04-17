#!/usr/bin/env crystal

# Embeddings example using llama.cr
#
# This example demonstrates how to use the embeddings functionality to:
# 1. Extract embeddings from text
# 2. Calculate similarity between texts
#
# Compilation:
#   crystal build examples/embeddings.cr --link-flags="-L/path/to/llama.cpp/build/bin"
#
# Execution:
#   LD_LIBRARY_PATH=/path/to/llama.cpp/build/bin ./embeddings /path/to/model.gguf

require "../src/llama"

# Check command line arguments
if ARGV.size < 1
  puts "Embeddings Example for llama.cr"
  puts
  puts "Compilation:"
  puts "  crystal build examples/embeddings.cr --link-flags=\"-L/path/to/llama.cpp/build/bin\""
  puts
  puts "Execution:"
  puts "  LD_LIBRARY_PATH=/path/to/llama.cpp/build/bin ./embeddings /path/to/model.gguf"
  puts
  puts "Parameters:"
  puts "  1. Path to model file (required)"
  exit 1
end

model_path = ARGV[0]

# Function to calculate cosine similarity between two vectors
def cosine_similarity(a : Array(Float32), b : Array(Float32)) : Float32
  # Check if vectors have the same dimension
  if a.size != b.size
    raise ArgumentError.new("Vectors must have the same dimension")
  end

  # Calculate dot product
  dot_product = 0.0_f32
  a.size.times do |i|
    dot_product += a[i] * b[i]
  end

  # Calculate magnitudes
  magnitude_a = Math.sqrt(a.sum { |x| x * x })
  magnitude_b = Math.sqrt(b.sum { |x| x * x })

  # Calculate cosine similarity
  if magnitude_a.zero? || magnitude_b.zero?
    0.0_f32
  else
    dot_product / (magnitude_a * magnitude_b)
  end
end

puts "Loading model from #{model_path}..."
model = Llama::Model.new(model_path)
puts "Model loaded successfully!"
puts "  - Parameters: #{model.n_params}"
puts "  - Embedding size: #{model.n_embd}"

# Create a context with embeddings enabled
puts "\nCreating context with embeddings enabled..."
context = model.context(embeddings: true)

# Sample texts to compare
texts = [
  "The quick brown fox jumps over the lazy dog.",
  "A fast auburn fox leaps above the sleepy canine.",
  "The weather is nice today.",
  "It's a beautiful sunny day outside.",
]

puts "\nExtracting embeddings for sample texts..."
embeddings = [] of Array(Float32)

texts.each do |text|
  puts "Processing: \"#{text}\""

  # Tokenize the text
  tokens = model.vocab.tokenize(text)

  # Create a batch for the tokens
  batch = Llama::Batch.get_one(tokens)

  # Process the batch
  context.decode(batch)

  # Get the embeddings
  emb = context.get_embeddings_seq(0)

  if emb.nil?
    puts "  - Failed to get embeddings"
  else
    puts "  - Got embedding vector of size #{emb.size}"
    embeddings << emb
  end
end

# Calculate similarities between all pairs of texts
puts "\nCalculating similarities between texts:"
texts.size.times do |i|
  (i + 1).upto(texts.size - 1) do |j|
    if i < embeddings.size && j < embeddings.size
      similarity = cosine_similarity(embeddings[i], embeddings[j])
      puts "Similarity between text #{i + 1} and text #{j + 1}: #{similarity.round(4)}"
    end
  end
end

# Print model metadata
puts "\nModel Metadata:"
metadata = model.metadata
metadata.each do |key, value|
  # Truncate very long values
  display_value = value.size > 100 ? "#{value[0, 97]}..." : value
  puts "  - #{key}: #{display_value}"
end

puts "\nModel description: #{model.description}"

puts "\nEmbeddings example completed!"
