require "./spec_helper"

describe Llama::Batch do
  describe ".new" do
    it "creates a batch with the specified parameters" do
      batch = Llama::Batch.new(10)
      batch.should_not be_nil
      batch.n_tokens.should eq(10)
    end

    it "raises an error with invalid parameters" do
      expect_raises(ArgumentError) do
        Llama::Batch.new(0)
      end

      expect_raises(ArgumentError) do
        Llama::Batch.new(-5)
      end
    end
  end

  describe ".get_one" do
    it "creates a batch from an array of tokens" do
      tokens = [1, 2, 3, 4, 5]
      batch = Llama::Batch.get_one(tokens)
      batch.should_not be_nil
      batch.n_tokens.should eq(tokens.size)
    end

    it "handles empty token arrays" do
      tokens = [] of Int32
      batch = Llama::Batch.get_one(tokens)
      batch.should_not be_nil
      batch.n_tokens.should eq(0)
    end
  end

  describe "#set_token" do
    it "sets a token at the specified index" do
      batch = Llama::Batch.new(5)
      # This test just ensures that set_token doesn't raise an error
      batch.set_token(0, 42)
      batch.set_token(1, 43, 1)
      batch.set_token(2, 44, 2, [1] of Int32)
      batch.set_token(3, 45, 3, [1, 2] of Int32, true)
      batch.set_token(4, 46, 4, [1, 2, 3] of Int32, false)
    end

    it "raises an error with invalid index" do
      batch = Llama::Batch.new(3)
      expect_raises(IndexError) do
        batch.set_token(-1, 42)
      end

      expect_raises(IndexError) do
        batch.set_token(3, 42)
      end
    end
  end

  # Model-dependent tests
  model_path = ENV["MODEL_PATH"]? || ARGV.find { |arg| arg.starts_with?("--model=") }.try &.split("=")[1]?

  if model_path.nil?
    pending "Skipping model-dependent batch tests (no model provided)"
  else
    it "works with a real model context" do
      model = Llama::Model.new(model_path)
      context = model.context

      # Create a batch with a simple token sequence
      tokens = model.vocab.tokenize("Hello, world!")
      batch = Llama::Batch.get_one(tokens)

      # Process the batch
      result = context.decode(batch)
      result.should be >= 0
    end
  end
end
