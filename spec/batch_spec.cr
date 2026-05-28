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

  describe "#clone_dup" do
    it "raises NotImplementedError when clone is called" do
      batch = Llama::Batch.new(10)
      expect_raises(NotImplementedError, "clone is not supported for Llama::Batch") do
        batch.clone
      end
    end

    it "raises NotImplementedError when dup is called" do
      batch = Llama::Batch.new(10)
      expect_raises(NotImplementedError, "dup is not supported for Llama::Batch") do
        batch.dup
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

    it "rejects empty token arrays" do
      tokens = [] of Int32
      expect_raises(ArgumentError, "Tokens array cannot be empty") do
        Llama::Batch.get_one(tokens)
      end
    end
  end

  describe ".from_tokens" do
    it "copies tokens into owned batch storage" do
      tokens = [10, 20, 30]
      batch = Llama::Batch.from_tokens(tokens)
      handle = batch.to_unsafe

      handle.n_tokens.should eq(tokens.size)
      tokens.each_with_index do |token, i|
        handle.token[i].should eq(token)
      end
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

    it "sets logits to false explicitly" do
      batch = Llama::Batch.new(1)

      batch.set_token(0, 42, logits: true)
      batch.to_unsafe.logits[0].should eq(1_i8)

      batch.set_token(0, 42, logits: false)
      batch.to_unsafe.logits[0].should eq(0_i8)
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

    it "raises an error for embedding batches" do
      batch = Llama::Batch.new(1, 4)

      expect_raises(ArgumentError, "Batch is not token-based") do
        batch.set_token(0, 42)
      end
    end
  end

  describe "#set_embedding" do
    it "sets logits to false explicitly" do
      batch = Llama::Batch.new(1, 4)

      batch.set_embedding(0, [0.0_f32, 1.0_f32, 2.0_f32, 3.0_f32], logits: true)
      batch.to_unsafe.logits[0].should eq(1_i8)

      batch.set_embedding(0, [0.0_f32, 1.0_f32, 2.0_f32, 3.0_f32], logits: false)
      batch.to_unsafe.logits[0].should eq(0_i8)
    end

    it "raises an error for token batches" do
      batch = Llama::Batch.new(1)

      expect_raises(ArgumentError, "Batch is not embedding-based") do
        batch.set_embedding(0, [0.0_f32])
      end
    end
  end

  it "works with a real model context" do
    model = Llama::Model.new(MODEL_PATH)
    context = model.context

    # Create a batch with a simple token sequence
    tokens = model.vocab.tokenize("Hello, world!")
    batch = Llama::Batch.from_tokens(tokens)

    # Process the batch
    result = context.decode(batch)
    result.should be >= 0
  end
end
