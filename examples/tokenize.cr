#!/usr/bin/env crystal

# Tokenization example using llama.cr
#
# This example demonstrates how to tokenize text using a model's vocabulary.
# It's a direct port of the C++ tokenize.cpp example from llama.cpp.
#
# Compilation:
#   crystal build examples/tokenize.cr --link-flags="-L/path/to/llama.cpp/build/bin"
#
# Execution:
#   LD_LIBRARY_PATH=/path/to/llama.cpp/build/bin ./tokenize [options]

require "../src/llama"
require "option_parser"

# Helper function to read prompt from a file
def read_prompt_from_file(filepath : String) : String
  begin
    File.read(filepath)
  rescue ex
    STDERR.puts "#{__FILE__}: could not open file '#{filepath}' for reading: #{ex.message}"
    exit(1)
  end
end

# Parse command line arguments
printing_ids = false
no_bos = false
no_escape = false
no_parse_special = false
disable_logging = false
show_token_count = false
model_path = nil
prompt_path = nil
prompt_arg = nil
stdin_set = false

# Track which arguments were explicitly given
model_path_set = false
prompt_path_set = false
prompt_set = false

OptionParser.parse do |parser|
  parser.banner = "Usage: #{PROGRAM_NAME} [options]"

  parser.on("-h", "--help", "Print this help and exit") do
    puts parser
  end

  parser.on("--ids", "Only print numerical token IDs, not token strings") do
    printing_ids = true
  end

  parser.on("-m MODEL_PATH", "--model=MODEL_PATH", "Path to model (required)") do |path|
    model_path = path
    model_path_set = true
  end

  parser.on("--no-bos", "Do not add BOS token to the prompt") do
    no_bos = true
  end

  parser.on("--no-escape", "Do not escape input (such as \\n, \\t, etc.)") do
    no_escape = true
  end

  parser.on("--no-parse-special", "Do not parse control tokens") do
    no_parse_special = true
  end

  parser.on("-p PROMPT", "--prompt=PROMPT", "Read prompt from the argument") do |prompt|
    prompt_arg = prompt
    prompt_set = true
  end

  parser.on("-f PROMPT_FNAME", "--file=PROMPT_FNAME", "Read prompt from a file") do |path|
    prompt_path = path
    prompt_path_set = true
  end

  parser.on("--stdin", "Read prompt from standard input") do
    stdin_set = true
  end

  parser.on("--log-disable", "Disable logs") do
    disable_logging = true
  end

  parser.on("--show-count", "Print the total number of tokens") do
    show_token_count = true
  end

  parser.invalid_option do |flag|
    STDERR.puts "Error: unknown option '#{flag}'"
    STDERR.puts "Run with --help for usage information."
    exit(1)
  end
end

# Sanity check the command line arguments
if !model_path_set || model_path.nil?
  STDERR.puts "Error: must specify --model."
  exit(1)
end

prompts_set = [prompt_path_set, prompt_set, stdin_set].count(true)
if prompts_set > 1
  STDERR.puts "Error: --stdin, --file and --prompt are mutually exclusive."
  exit(1)
end

if prompts_set == 0
  STDERR.puts "Error: must specify one of: --stdin, --file or --prompt."
  exit(1)
end

# Figure out where the prompt will come from
prompt = ""
if prompt_path_set && !prompt_path.nil?
  prompt = read_prompt_from_file(prompt_path.not_nil!)
elsif prompt_set && !prompt_arg.nil?
  prompt = prompt_arg
else
  # We'll read stdin after loading the model
end

# Start actually doing the tokenizing stuff
if disable_logging
  Llama::LibLlama.llama_log_set(nil, nil)
end

Llama::LibLlama.llama_backend_init

begin
  # Load the model with vocab_only=true
  model = Llama::Model.new(model_path.not_nil!, vocab_only: true)
rescue ex
  STDERR.puts "Error: could not load model from file '#{model_path}': #{ex.message}"
  exit(1)
end

vocab = model.vocab

# Create a minimal context (needed for token_to_piece)
ctx_params = Llama::LibLlama.llama_context_default_params
begin
  ctx = model.context(ctx_params)
rescue ex
  STDERR.puts "Error: could not create context: #{ex.message}"
  exit(1)
end

# Read entire prompt from stdin?
if stdin_set
  begin
    prompt = STDIN.gets_to_end
  rescue ex
    STDERR.puts "Error: could not read the entire standard input: #{ex.message}"
    exit(1)
  end
end

# Process escape sequences if needed
if !no_escape
  prompt = Llama.process_escapes(prompt.not_nil!)
end

# Tokenize the prompt
model_wants_add_bos = vocab.add_bos?
add_bos = model_wants_add_bos && !no_bos
parse_special = !no_parse_special

begin
  tokens = vocab.tokenize(prompt.not_nil!, add_bos, parse_special)
rescue ex
  STDERR.puts "Error: failed to tokenize prompt: #{ex.message}"
  exit(1)
end

# Print the tokens
if printing_ids
  print "["
  tokens.each_with_index do |token, i|
    print ", " if i > 0
    print token
  end
  puts "]"
else
  tokens.each do |token|
    begin
      puts vocab.format_token(token)
    rescue ex
      puts "#{token} -> (utf-8 decode failure)"
    end
  end
end

# Show token count if requested
if show_token_count
  puts "Total number of tokens: #{tokens.size}"
end
