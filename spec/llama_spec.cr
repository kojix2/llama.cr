require "./spec_helper"

describe Llama do
  describe ".system_info" do
    it "returns system information as a string" do
      info = Llama.system_info
      info.should be_a(String)
      info.should_not be_empty
    end
  end

  # These tests require a model file and are commented out by default
  # Uncomment and modify with a valid model path to run them

  # describe ".generate" do
  #   it "generates text from a prompt" do
  #     model_path = "/path/to/model.gguf"
  #     prompt = "Once upon a time"
  #
  #     response = Llama.generate(model_path, prompt, max_tokens: 10)
  #     response.should be_a(String)
  #     response.should_not be_empty
  #   end
  # end

  # describe "Model" do
  #   it "loads a model" do
  #     model_path = "/path/to/model.gguf"
  #     model = Llama::Model.new(model_path)
  #
  #     model.n_params.should be > 0
  #     model.n_embd.should be > 0
  #     model.n_layer.should be > 0
  #     model.n_head.should be > 0
  #   end
  #
  #   it "gets the vocabulary" do
  #     model_path = "/path/to/model.gguf"
  #     model = Llama::Model.new(model_path)
  #
  #     vocab = model.vocab
  #     vocab.should be_a(Llama::Vocab)
  #     vocab.n_tokens.should be > 0
  #   end
  # end

  # describe "Context" do
  #   it "creates a context from a model" do
  #     model_path = "/path/to/model.gguf"
  #     model = Llama::Model.new(model_path)
  #
  #     context = model.context
  #     context.should be_a(Llama::Context)
  #   end
  # end

  # describe "Vocab" do
  #   it "tokenizes text" do
  #     model_path = "/path/to/model.gguf"
  #     model = Llama::Model.new(model_path)
  #     vocab = model.vocab
  #
  #     tokens = vocab.tokenize("Hello, world!")
  #     tokens.should be_a(Array(Int32))
  #     tokens.size.should be > 0
  #   end
  # end
end
