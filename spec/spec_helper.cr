require "spec"
require "../src/llama"

Llama.log_level = Llama::LOG_LEVEL_ERROR

MODEL_PATH = ENV["MODEL_PATH"] || raise "MODEL_PATH environment variable not set"
unless File.exists?(MODEL_PATH)
  raise "Model file not found: #{MODEL_PATH}"
end
