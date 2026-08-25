require "../src/llama"
require "option_parser"

# Helper function to read prompt from a file
def read_prompt_from_file(filepath : String) : String
  File.read(filepath)
rescue ex
  STDERR.puts "#{__FILE__}: could not open file '#{filepath}' for reading: #{ex.message}"
  exit(1)
end

# Parse command line arguments
printing_ids = false
no_bos = false
no_escape = false
no_parse_special = false
disable_logging = false
show_token_count = false
model_path = ""
prompt_path = ""
prompt_arg = ""
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
if !model_path_set || model_path.empty?
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
if prompt_path_set
  prompt = read_prompt_from_file(prompt_path)
elsif prompt_set
  prompt = prompt_arg
else
  # We'll read stdin after loading the model
end

# Start actually doing the tokenizing stuff
if disable_logging
  Llama::LibLlama.llama_log_set(nil, nil)
end

begin
  Llama::Model.open(model_path, vocab_only: true) do |model|
    vocab = model.vocab

    if stdin_set
      begin
        prompt = STDIN.gets_to_end
      rescue ex
        raise "could not read the entire standard input: #{ex.message}"
      end
    end

    prompt = Llama.process_escapes(prompt) unless no_escape

    model_wants_add_bos = vocab.add_bos?
    add_bos = model_wants_add_bos && !no_bos
    parse_special = !no_parse_special

    begin
      tokens = vocab.tokenize(prompt, add_bos, parse_special)
    rescue ex
      raise "failed to tokenize prompt: #{ex.message}"
    end

    if printing_ids
      print "["
      tokens.each_with_index do |token, i|
        print ", " if i > 0
        print token
      end
      puts "]"
    else
      tokens.each do |token|
        puts vocab.format_token(token)
      rescue
        puts "#{token} -> (utf-8 decode failure)"
      end
    end

    if show_token_count
      puts "Total number of tokens: #{tokens.size}"
    end
  end
rescue ex
  STDERR.puts "Error: #{ex.message}"
  exit(1)
end
