# spec/context_lora_spec.cr
require "../src/llama"

describe Llama::Context do
  MODEL_PATH   = ENV["LLAMA_TEST_MODEL"]? || "models/tinyllama.gguf"
  ADAPTER_PATH = ENV["LLAMA_TEST_ADAPTER"]? || "models/dummy_adapter.bin"

  it "can attach and detach a LoRA adapter" do
    skip "Test model or adapter file not found" unless File.exists?(MODEL_PATH) && File.exists?(ADAPTER_PATH)
    model = Llama::Model.new(MODEL_PATH)
    context = Llama::Context.new(model)
    adapter = Llama::AdapterLora.new(model, ADAPTER_PATH)

    context.attach_adapter_lora(adapter).should eq 0
    context.detach_adapter_lora(adapter).should eq 0
  end

  it "can clear all LoRA adapters" do
    skip "Test model or adapter file not found" unless File.exists?(MODEL_PATH) && File.exists?(ADAPTER_PATH)
    model = Llama::Model.new(MODEL_PATH)
    context = Llama::Context.new(model)
    adapter = Llama::AdapterLora.new(model, ADAPTER_PATH)

    context.attach_adapter_lora(adapter).should eq 0
    context.clear_adapters_lora
    # No exception means success
  end

  it "raises error when attaching an invalid adapter" do
    skip "Test model file not found" unless File.exists?(MODEL_PATH)
    model = Llama::Model.new(MODEL_PATH)
    context = Llama::Context.new(model)
    # Use a dummy object to simulate invalid adapter
    expect_raises(Exception) do
      context.attach_adapter_lora("not_an_adapter".as(Llama::AdapterLora))
    end
  end

  it "can apply a dummy control vector" do
    skip "Test model or adapter file not found" unless File.exists?(MODEL_PATH) && File.exists?(ADAPTER_PATH)
    model = Llama::Model.new(MODEL_PATH)
    context = Llama::Context.new(model)
    adapter = Llama::AdapterLora.new(model, ADAPTER_PATH)
    context.attach_adapter_lora(adapter).should eq 0

    n_embd = model.n_embd
    n_layers = 1
    data = Slice(Float32).new(n_embd * n_layers, 0.0)
    # Should not raise
    context.apply_adapter_cvec(data, n_embd, 1, 1)
  end
end
