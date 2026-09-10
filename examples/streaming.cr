require "../src/llama"
require "option_parser"

model_path = ""
prompt = "Once upon a time"

OptionParser.parse do |parser|
  parser.banner = "Usage: #{PROGRAM_NAME} --model MODEL [--prompt TEXT]"
  parser.on("-m", "--model MODEL", "Path to a GGUF model") { |path| model_path = path }
  parser.on("-p", "--prompt TEXT", "Prompt to complete") { |text| prompt = text }
  parser.on("-h", "--help", "Show this help") { puts parser; exit }
end

abort "--model is required" if model_path.empty?

Llama::Model.open(model_path) do |model|
  model.session do |session|
    result = session.generate(prompt) do |chunk|
      print chunk.text
      STDOUT.flush
    end
    puts "\n[#{result.finish_reason}; #{result.usage.generated_tokens} tokens]"
  end
end
