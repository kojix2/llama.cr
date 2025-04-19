require "../src/llama"
require "option_parser"

model_path = ""
prompt = "Hello my name is"
max_tokens = 100

OptionParser.parse do |parser|
  parser.on("-m", "--model=MODEL", "Path to the model file") { |m| model_path = m }
  parser.on("-p", "--prompt=PROMPT", "Prompt text") { |p| prompt = p }
  parser.on("-n", "--max-tokens=N", "Number of tokens to generate") { |n| max_tokens = n.to_i }
  parser.on("-h", "--help", "Show this help") { puts parser; exit }
end

if model_path.empty?
  STDERR.puts "Please specify the model file path with -m"
  exit 1
end

Llama.log_level = Llama::LOG_LEVEL_ERROR

result = Llama.generate(model_path, prompt, max_tokens: max_tokens)
puts
puts prompt
puts result
