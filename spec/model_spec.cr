require "./spec_helper"

# This file contains tests that require an actual model file
# Run with: crystal spec spec/model_spec.cr -- --model=/path/to/model.gguf

describe "Llama with model" do
  # Get model path from command line arguments or environment variable
  model_path = ENV["MODEL_PATH"]? || ARGV.find { |arg| arg.starts_with?("--model=") }.try &.split("=")[1]?

  if model_path.nil?
    pending "Skipping model tests (no model provided)"
  else
    it "loads the model" do
      model = Llama::Model.new(model_path)
      model.should_not be_nil
      model.n_params.should be > 0
      puts "  - Model parameters: #{model.n_params}"
      puts "  - Embedding size: #{model.n_embd}"
      puts "  - Layers: #{model.n_layer}"
      puts "  - Attention heads: #{model.n_head}"
    end

    it "checks model architecture properties" do
      model = Llama::Model.new(model_path)

      # Test encoder/decoder properties
      puts "  - Has encoder: #{model.has_encoder?}"
      puts "  - Has decoder: #{model.has_decoder?}"

      # Test if model is recurrent
      puts "  - Is recurrent: #{model.recurrent?}"

      # Test RoPE frequency scaling factor
      puts "  - RoPE frequency scaling factor: #{model.rope_freq_scale_train}"

      # Test decoder start token
      decoder_start_token = model.decoder_start_token
      puts "  - Decoder start token: #{decoder_start_token}"

      # These are informational tests, not assertions
      # We just want to make sure the methods don't crash
      model.has_encoder?.should be_a(Bool)
      model.has_decoder?.should be_a(Bool)
      model.recurrent?.should be_a(Bool)
      model.rope_freq_scale_train.should be_a(Float32)
      model.decoder_start_token.should be_a(Int32)
    end

    it "accesses the vocabulary" do
      model = Llama::Model.new(model_path)
      vocab = model.vocab
      vocab.should_not be_nil
      vocab.n_tokens.should be > 0
      puts "  - Vocabulary size: #{vocab.n_tokens}"
    end

    it "tokenizes text" do
      model = Llama::Model.new(model_path)
      vocab = model.vocab

      text = "Hello, world!"
      tokens = vocab.tokenize(text)
      tokens.should be_a(Array(Int32))
      tokens.size.should be > 0
      puts "  - Tokenized '#{text}' to #{tokens.size} tokens: #{tokens.inspect}"
    end

    it "creates a context" do
      model = Llama::Model.new(model_path)
      context = model.context
      context.should_not be_nil
    end

    it "generates text" do
      prompt = "Hello, my name is"
      response = Llama.generate(model_path, prompt, max_tokens: 10)
      response.should be_a(String)
      response.should_not be_empty
      puts "  - Prompt: '#{prompt}'"
      puts "  - Response: '#{response}'"
    end
  end
end
