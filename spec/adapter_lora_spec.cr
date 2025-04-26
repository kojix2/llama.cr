require "./spec_helper"

describe "Llama::AdapterLora" do
  it "successfully loads a valid LoRA adapter" do
    pending! "Test model or adapter file not found" unless File.exists?(ADAPTER_PATH)
    model = Llama::Model.new(MODEL_PATH)
    adapter = Llama::AdapterLora.new(model, ADAPTER_PATH)
    adapter.to_unsafe.should_not be_nil
  end

  it "raises AdapterLora::Error for a non-existent adapter file" do
    pending! "Test model or adapter file not found" unless File.exists?(ADAPTER_PATH)
    model = Llama::Model.new(MODEL_PATH)
    expect_raises(Llama::AdapterLora::Error) do
      Llama::AdapterLora.new(model, "no_such_file.bin")
    end
  end

  it "raises NotImplementedError for clone" do
    pending! "Test model or adapter file not found" unless File.exists?(ADAPTER_PATH)
    model = Llama::Model.new(MODEL_PATH)
    adapter = Llama::AdapterLora.new(model, ADAPTER_PATH)
    expect_raises(NotImplementedError) { adapter.clone }
  end

  it "raises NotImplementedError for dup" do
    pending! "Test model or adapter file not found" unless File.exists?(ADAPTER_PATH)
    model = Llama::Model.new(MODEL_PATH)
    adapter = Llama::AdapterLora.new(model, ADAPTER_PATH)
    expect_raises(NotImplementedError) { adapter.dup }
  end
end
