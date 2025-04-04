#!/usr/bin/env crystal

# Chat example using llama.cr
#
# Compilation:
#   crystal build examples/chat.cr --link-flags="-L/path/to/llama.cpp/build/bin"
#
# Execution:
#   LD_LIBRARY_PATH=/path/to/llama.cpp/build/bin ./chat /path/to/model.gguf

require "../src/llama"
require "../src/llama/chat"

# Check command line arguments
if ARGV.size < 1
  puts "Chat Example for llama.cr"
  puts
  puts "Compilation:"
  puts "  crystal build examples/chat.cr --link-flags=\"-L/path/to/llama.cpp/build/bin\""
  puts
  puts "Execution:"
  puts "  LD_LIBRARY_PATH=/path/to/llama.cpp/build/bin ./chat /path/to/model.gguf"
  puts
  puts "Parameters:"
  puts "  1. Path to model file (required)"
  exit 1
end

model_path = ARGV[0]

puts "Loading model from #{model_path}..."
model = Llama::Model.new(model_path)
puts "Model loaded successfully!"

puts "\nCreating context..."
context = model.context
puts "Context created successfully!"

# Check if the model has a chat template
template = model.chat_template
if template
  puts "\nModel has a built-in chat template"
else
  puts "\nModel does not have a built-in chat template, using default"
end

# Create a simple chat conversation
messages = [
  Llama::ChatMessage.new("system", "You are a helpful assistant."),
  Llama::ChatMessage.new("user", "Hello, who are you?"),
]

puts "\nGenerating chat response..."
start_time = Time.monotonic
response = context.chat(messages)
end_time = Time.monotonic

puts "\nChat:"
puts "System: You are a helpful assistant."
puts "User: Hello, who are you?"
puts "Assistant: #{response}"
puts "\nGeneration took #{(end_time - start_time).total_seconds.round(2)} seconds"

# Check for non-interactive mode
if ENV["NON_INTERACTIVE"] != "true"
  # Interactive chat mode
  puts "\nEnter 'exit' to quit"
  loop do
    print "\nYou: "
    user_input = gets
    break if user_input.nil? || user_input.downcase == "exit"

    messages << Llama::ChatMessage.new("user", user_input)

    print "Assistant: "
    start_time = Time.monotonic
    response = context.chat(messages)
    end_time = Time.monotonic

    puts response
    puts "(Generated in #{(end_time - start_time).total_seconds.round(2)} seconds)"

    messages << Llama::ChatMessage.new("assistant", response)
  end
end
