require "spec"
require "../src/llama"

Llama.log_level = Llama::LOG_LEVEL_ERROR

MODEL_PATH = ENV["MODEL_PATH"]
unless File.exists?(MODEL_PATH)
  raise "Model file not found: #{MODEL_PATH}"
end

ADAPTER_PATH = ENV.fetch("ADAPTER_PATH", "")
if ADAPTER_PATH != ""
  unless File.exists?(ADAPTER_PATH)
    raise "Adapter file not found: #{ADAPTER_PATH}"
  end
end
