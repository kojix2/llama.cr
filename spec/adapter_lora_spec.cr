require "./spec_helper"

lib LibAdapterSpecStdio
  fun fopen(path : LibC::Char*, mode : LibC::Char*) : Llama::LibLlama::CFile*
  fun fclose(file : Llama::LibLlama::CFile*) : Int32
end

describe "Llama::AdapterLora" do
  it "loads through a caller-owned C FILE pointer" do
    pending! "Test model or adapter file not found" unless File.exists?(ADAPTER_PATH)
    Llama::Model.open(MODEL_PATH) do |model|
      file = LibAdapterSpecStdio.fopen(ADAPTER_PATH, "rb")
      file.null?.should be_false
      begin
        adapter = Llama::LibLlama.llama_adapter_lora_init_from_file_ptr(model.unsafe_handle!, file)
        adapter.null?.should be_false
        Llama::LibLlama.llama_adapter_meta_count(adapter).should be > 0
      ensure
        Llama::LibLlama.llama_adapter_lora_free(adapter) if adapter && !adapter.null?
        LibAdapterSpecStdio.fclose(file) unless file.null?
      end
    end
  end

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
