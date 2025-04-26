# spec/adapter_lora_spec.cr
require "../src/llama"

describe Llama::AdapterLora do
  # Set these paths to valid files for your environment
  MODEL_PATH   = ENV["LLAMA_TEST_MODEL"]? || "models/tinyllama.gguf"
  ADAPTER_PATH = ENV["LLAMA_TEST_ADAPTER"]? || "models/dummy_adapter.bin"

  describe "#initialize" do
    it "successfully loads a valid LoRA adapter" do
      skip "Test model or adapter file not found" unless File.exists?(MODEL_PATH) && File.exists?(ADAPTER_PATH)
      model = Llama::Model.new(MODEL_PATH)
      adapter = Llama::AdapterLora.new(model, ADAPTER_PATH)
      adapter.to_unsafe.should_not be_nil
    end

    it "raises AdapterLora::Error for a non-existent adapter file" do
      skip "Test model file not found" unless File.exists?(MODEL_PATH)
      model = Llama::Model.new(MODEL_PATH)
      expect_raises(Llama::AdapterLora::Error) do
        Llama::AdapterLora.new(model, "no_such_file.bin")
      end
    end
  end

  describe "#clone and #dup" do
    it "raises NotImplementedError for clone" do
      skip "Test model or adapter file not found" unless File.exists?(MODEL_PATH) && File.exists?(ADAPTER_PATH)
      model = Llama::Model.new(MODEL_PATH)
      adapter = Llama::AdapterLora.new(model, ADAPTER_PATH)
      expect_raises(NotImplementedError) { adapter.clone }
    end

    it "raises NotImplementedError for dup" do
      skip "Test model or adapter file not found" unless File.exists?(MODEL_PATH) && File.exists?(ADAPTER_PATH)
      model = Llama::Model.new(MODEL_PATH)
      adapter = Llama::AdapterLora.new(model, ADAPTER_PATH)
      expect_raises(NotImplementedError) { adapter.dup }
    end
  end
end
