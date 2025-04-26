require "./spec_helper"

describe "Llama::AdapterLora" do
  adapter_path = ENV["ADAPTER_PATH"]

  it "successfully loads a valid LoRA adapter" do
    model = Llama::Model.new(MODEL_PATH)
    adapter = Llama::AdapterLora.new(model, adapter_path)
    adapter.to_unsafe.should_not be_nil
  end

  it "raises AdapterLora::Error for a non-existent adapter file" do
    model = Llama::Model.new(MODEL_PATH)
    expect_raises(Llama::AdapterLora::Error) do
      Llama::AdapterLora.new(model, "no_such_file.bin")
    end
  end

  it "raises NotImplementedError for clone" do
    model = Llama::Model.new(MODEL_PATH)
    adapter = Llama::AdapterLora.new(model, adapter_path)
    expect_raises(NotImplementedError) { adapter.clone }
  end

  it "raises NotImplementedError for dup" do
    model = Llama::Model.new(MODEL_PATH)
    adapter = Llama::AdapterLora.new(model, adapter_path)
    expect_raises(NotImplementedError) { adapter.dup }
  end
end
