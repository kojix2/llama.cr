require "./spec_helper"

describe "Llama::AdapterLora" do
  model_path = ENV["MODEL_PATH"]
  adapter_path = ENV["ADAPTER_PATH"]

  if model_path.nil? || adapter_path.nil?
    pending "Skipping adapter lora tests (MODEL_PATH or ADAPTER_PATH not set)"
    next
  end

  it "successfully loads a valid LoRA adapter" do
    model = Llama::Model.new(model_path)
    adapter = Llama::AdapterLora.new(model, adapter_path)
    adapter.to_unsafe.should_not be_nil
  end

  it "raises AdapterLora::Error for a non-existent adapter file" do
    model = Llama::Model.new(model_path)
    expect_raises(Llama::AdapterLora::Error) do
      Llama::AdapterLora.new(model, "no_such_file.bin")
    end
  end

  it "raises NotImplementedError for clone" do
    model = Llama::Model.new(model_path)
    adapter = Llama::AdapterLora.new(model, adapter_path)
    expect_raises(NotImplementedError) { adapter.clone }
  end

  it "raises NotImplementedError for dup" do
    model = Llama::Model.new(model_path)
    adapter = Llama::AdapterLora.new(model, adapter_path)
    expect_raises(NotImplementedError) { adapter.dup }
  end
end
