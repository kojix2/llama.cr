require "./spec_helper"

describe Llama::Context do
  it "rejects closing an attached LoRA adapter" do
    pending! "Test model or adapter file not found" unless File.exists?(ADAPTER_PATH)
    model = Llama::Model.new(MODEL_PATH)
    context = model.context
    adapter = Llama::AdapterLora.new(model, ADAPTER_PATH)
    context.attach_adapter_lora(adapter)

    expect_raises(Llama::BusyError, "Llama::AdapterLora is busy") { adapter.close }

    context.close
    adapter.close
    model.close
  end

  it "can attach and detach a LoRA adapter" do
    pending! "Test model or adapter file not found" unless File.exists?(ADAPTER_PATH)
    model = Llama::Model.new(MODEL_PATH)
    context = Llama::Context.new(model)
    adapter = Llama::AdapterLora.new(model, ADAPTER_PATH)

    context.attach_adapter_lora(adapter).should eq 0
    context.detach_adapter_lora(adapter).should eq 0
  end

  it "can clear all LoRA adapters" do
    pending! "Test model or adapter file not found" unless File.exists?(ADAPTER_PATH)
    model = Llama::Model.new(MODEL_PATH)
    context = Llama::Context.new(model)
    adapter = Llama::AdapterLora.new(model, ADAPTER_PATH)

    context.attach_adapter_lora(adapter).should eq 0
    context.clear_adapters_lora
    # No exception means success
  end

  it "rejects a LoRA adapter loaded for another model" do
    pending! "Test model or adapter file not found" unless File.exists?(ADAPTER_PATH)
    model = Llama::Model.new(MODEL_PATH)
    other_model = Llama::Model.new(MODEL_PATH)
    context = Llama::Context.new(model)
    adapter = Llama::AdapterLora.new(other_model, ADAPTER_PATH)

    expect_raises(Llama::Context::Error, "LoRA adapter was loaded for a different model") do
      context.attach_adapter_lora(adapter)
    end
  end

  it "can apply a dummy control vector" do
    pending! "Test model or adapter file not found" unless File.exists?(ADAPTER_PATH)
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
