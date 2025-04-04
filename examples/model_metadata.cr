#!/usr/bin/env crystal

# Model metadata example using llama.cr
#
# This example demonstrates how to access and use model metadata:
# 1. Get metadata values by key
# 2. List all metadata entries
# 3. Get model description
#
# Compilation:
#   crystal build examples/model_metadata.cr --link-flags="-L/path/to/llama.cpp/build/bin"
#
# Execution:
#   LD_LIBRARY_PATH=/path/to/llama.cpp/build/bin ./model_metadata /path/to/model.gguf

require "../src/llama"

# Check command line arguments
if ARGV.size < 1
  puts "Model Metadata Example for llama.cr"
  puts
  puts "Compilation:"
  puts "  crystal build examples/model_metadata.cr --link-flags=\"-L/path/to/llama.cpp/build/bin\""
  puts
  puts "Execution:"
  puts "  LD_LIBRARY_PATH=/path/to/llama.cpp/build/bin ./model_metadata /path/to/model.gguf"
  puts
  puts "Parameters:"
  puts "  1. Path to model file (required)"
  exit 1
end

model_path = ARGV[0]

puts "Loading model from #{model_path}..."
model = Llama::Model.new(model_path)
puts "Model loaded successfully!"

# Print basic model information
puts "\nBasic Model Information:"
puts "  - Parameters: #{model.n_params}"
puts "  - Embedding size: #{model.n_embd}"
puts "  - Layers: #{model.n_layer}"
puts "  - Attention heads: #{model.n_head}"

# Get model description
puts "\nModel Description:"
puts "  #{model.description}"

# Get the number of metadata entries
meta_count = model.meta_count
puts "\nMetadata Count: #{meta_count}"

# Get specific metadata values
common_keys = ["general.architecture", "general.name", "llama.context_length", "tokenizer.ggml.model"]

puts "\nCommon Metadata Values:"
common_keys.each do |key|
  value = model.meta_val_str(key)
  if value
    # Truncate very long values
    display_value = value.size > 100 ? "#{value[0, 97]}..." : value
    puts "  - #{key}: #{display_value}"
  else
    puts "  - #{key}: Not found"
  end
end

# List all metadata entries
puts "\nAll Metadata Entries:"
metadata = model.metadata
metadata.each do |key, value|
  # Truncate very long values
  display_value = value.size > 100 ? "#{value[0, 97]}..." : value
  puts "  - #{key}: #{display_value}"
end

# Demonstrate accessing metadata by index
puts "\nAccessing Metadata by Index:"
3.times do |i|
  if i < meta_count
    key = model.meta_key_by_index(i)
    value = model.meta_val_str_by_index(i)

    if key && value
      # Truncate very long values
      display_value = value.size > 100 ? "#{value[0, 97]}..." : value
      puts "  - Index #{i}: #{key} = #{display_value}"
    end
  end
end

# Check for chat template
chat_template = model.chat_template
puts "\nChat Template:"
if chat_template
  # Show just the first few lines of the template
  template_preview = chat_template.split("\n")[0, 5].join("\n")
  puts "  #{template_preview}..."
else
  puts "  No chat template found"
end

puts "\nModel metadata example completed!"
