require "./spec_helper"

describe Llama::Context do
  model_path = ENV["LLAMA_TEST_MODEL"]? || "models/tinyllama.gguf"
  adapter_path = ENV["LLAMA_TEST_ADAPTER"]? || "models/dummy_adapter.bin"

  it "can attach and detach a LoRA adapter" do
    pending! "Test model or adapter file not found" unless File.exists?(model_path) && File.exists?(adapter_path)
    model = Llama::Model.new(model_path)
    context = Llama::Context.new(model)
    adapter = Llama::AdapterLora.new(model, adapter_path)

    context.attach_adapter_lora(adapter).should eq 0
    context.detach_adapter_lora(adapter).should eq 0
  end

  it "can clear all LoRA adapters" do
    pending! "Test model or adapter file not found" unless File.exists?(model_path) && File.exists?(adapter_path)
    model = Llama::Model.new(model_path)
    context = Llama::Context.new(model)
    adapter = Llama::AdapterLora.new(model, adapter_path)

    context.attach_adapter_lora(adapter).should eq 0
    context.clear_adapters_lora
    # No exception means success
  end

  it "can apply a dummy control vector" do
    pending! "Test model or adapter file not found" unless File.exists?(model_path) && File.exists?(adapter_path)
    model = Llama::Model.new(model_path)
    context = Llama::Context.new(model)
    adapter = Llama::AdapterLora.new(model, adapter_path)
    context.attach_adapter_lora(adapter).should eq 0

    n_embd = model.n_embd
    n_layers = 1
    data = Slice(Float32).new(n_embd * n_layers, 0.0)
    # Should not raise
    context.apply_adapter_cvec(data, n_embd, 1, 1)
  end
end
