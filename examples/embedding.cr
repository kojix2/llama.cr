require "../src/llama"
require "option_parser"

model_path = ""
texts = [] of String

OptionParser.parse do |parser|
  parser.banner = "Usage: #{PROGRAM_NAME} --model MODEL TEXT..."
  parser.on("-m", "--model MODEL", "Path to an embedding-capable GGUF model") { |path| model_path = path }
  parser.on("-h", "--help", "Show this help") { puts parser; exit }
  parser.unknown_args { |args| texts = args }
end

abort "--model is required" if model_path.empty?
texts = ["Hello, world!"] if texts.empty?

Llama::Model.open(model_path) do |model|
  model.embedder(pooling: Llama::Pooling::Mean) do |embedder|
    embedder.embed_all(texts, normalize: true).each_with_index do |vector, index|
      puts "#{index}: #{vector.size} dimensions, first values: #{vector.first(5)}"
    end
  end
end
