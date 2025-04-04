#!/usr/bin/env crystal

# Tokenization and vocabulary example using llama.cr
#
# Compilation:
#   crystal build examples/tokenization.cr --link-flags="-L/path/to/llama.cpp/build/bin"
#
# Execution:
#   LD_LIBRARY_PATH=/path/to/llama.cpp/build/bin ./tokenization /path/to/model.gguf "Text to tokenize"

require "../src/llama"

# Check command line arguments
if ARGV.size < 2
  puts "Tokenization Example for llama.cr"
  puts
  puts "Compilation:"
  puts "  crystal build examples/tokenization.cr --link-flags=\"-L/path/to/llama.cpp/build/bin\""
  puts
  puts "Execution:"
  puts "  LD_LIBRARY_PATH=/path/to/llama.cpp/build/bin ./tokenization /path/to/model.gguf \"Text to tokenize\""
  puts
  puts "Parameters:"
  puts "  1. Path to model file (required)"
  puts "  2. Text to tokenize (required)"
  puts
  puts "Example:"
  puts "  ./tokenization tiny_model.gguf \"Hello, world!\""
  exit 1
end

model_path = ARGV[0]
text = ARGV[1]

puts "Loading model from #{model_path}..."
model = Llama::Model.new(model_path)
puts "Model loaded successfully!"

# Get the vocabulary
vocab = model.vocab
puts "Vocabulary size: #{vocab.n_tokens} tokens"

# Tokenize the text
puts "\nTokenizing text: '#{text}'"
tokens = vocab.tokenize(text)

puts "Text was tokenized into #{tokens.size} tokens:"
puts "Token IDs: #{tokens.inspect}"

# Convert tokens back to text
puts "\nConverting tokens back to text:"
tokens.each_with_index do |token, i|
  token_text = vocab.token_to_text(token)
  puts "Token #{i}: ID=#{token} Text='#{token_text}'"
end

# Demonstrate special tokens
puts "\nSpecial tokens in the vocabulary:"
special_tokens = {
  "BOS (Beginning of Sentence)": vocab.bos,
  "EOS (End of Sentence)":       vocab.eos,
  "EOT (End of Turn)":           vocab.eot,
  "NL (New Line)":               vocab.nl,
}

special_tokens.each do |name, token|
  if token >= 0
    puts "#{name}: ID=#{token} Text='#{vocab.token_to_text(token)}'"
  else
    puts "#{name}: Not defined in this model"
  end
end
