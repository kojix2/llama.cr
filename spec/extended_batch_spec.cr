require "./spec_helper"

describe "llama.cpp extended batch API" do
  it "decodes tokens, shares sequences, and reuses a cleared batch" do
    Llama::Model.open(MODEL_PATH) do |model|
      params = Llama::LibLlama.llama_context_default_params
      params.n_seq_max = 2
      params.kv_unified = true
      context = Llama::LibLlama.llama_init_from_model(model.unsafe_handle!, params)
      begin
        context.null?.should be_false
        batch = Llama::LibLlama.llama_batch_ext_init(context)
        batch.null?.should be_false
        tokens = model.vocab.tokenize("Hello")
        tokens.each_with_index do |token, i|
          Llama::LibLlama.llama_batch_ext_add_token(batch, 0, token).should eq(i)
          Llama::LibLlama.llama_batch_ext_add_seq(batch, i, 1).should be_true
          pos = i.to_i32
          Llama::LibLlama.llama_batch_ext_set_pos(batch, i, pointerof(pos)).should be_true
          Llama::LibLlama.llama_batch_ext_set_output_logits(batch, i, i == tokens.size - 1).should be_true
        end
        Llama::LibLlama.llama_process(context, Llama::LibLlama::LlamaProcessType::DECODE, batch).should eq(0)
        memory = Llama::LibLlama.llama_get_memory(context)
        Llama::LibLlama.llama_memory_seq_pos_max(memory, 0).should eq(tokens.size - 1)
        Llama::LibLlama.llama_memory_seq_pos_max(memory, 1).should eq(tokens.size - 1)

        Llama::LibLlama.llama_batch_ext_clear(batch)
        Llama::LibLlama.llama_batch_ext_add_token(batch, 0, tokens.last).should eq(0)
        pos = tokens.size.to_i32
        Llama::LibLlama.llama_batch_ext_set_pos(batch, 0, pointerof(pos)).should be_true
        Llama::LibLlama.llama_batch_ext_set_output_logits(batch, 0, true).should be_true
        Llama::LibLlama.llama_process(context, Llama::LibLlama::LlamaProcessType::DECODE, batch).should eq(0)
        Llama::LibLlama.llama_memory_seq_pos_max(memory, 0).should eq(tokens.size)
      ensure
        Llama::LibLlama.llama_batch_ext_free(batch) if batch && !batch.null?
        Llama::LibLlama.llama_free(context) unless context.null?
      end
    end
  end

  it "copies input embeddings and rejects invalid sequence IDs" do
    Llama::Model.open(MODEL_PATH) do |model|
      model.context do |context|
        batch = Llama::LibLlama.llama_batch_ext_init(context.unsafe_handle!)
        Llama::LibLlama.llama_batch_ext_add(batch, -1).should eq(-3)
        data = Array(Float32).new(model.n_embd_inp, 0.0_f32)
        embd = Llama::LibLlama::LlamaEmbd.new(data: data.to_unsafe, n_rows: 1, n_embd: data.size)
        Llama::LibLlama.llama_batch_ext_add_embd(batch, 0, embd).should eq(0)
        data.fill(Float32::NAN)
        pos = 0_i32
        Llama::LibLlama.llama_batch_ext_set_pos(batch, 0, pointerof(pos)).should be_true
        Llama::LibLlama.llama_batch_ext_set_output_embd(batch, 0, true).should be_true
        Llama::LibLlama.llama_process(context.unsafe_handle!, Llama::LibLlama::LlamaProcessType::DECODE, batch).should eq(0)
        Slice.new(context.logits, model.vocab.n_tokens).all?(&.finite?).should be_true
      ensure
        Llama::LibLlama.llama_batch_ext_free(batch) if batch && !batch.null?
      end
    end
  end
end
