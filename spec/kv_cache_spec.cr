require "./spec_helper"

describe Llama::KvCache do
  describe "#clone_dup" do
    it "raises NotImplementedError when clone is called" do
      model = Llama::Model.new(MODEL_PATH)
      context = model.context
      kv_cache = context.kv_cache
      expect_raises(NotImplementedError, "clone is not supported for Llama::KvCache") do
        kv_cache.clone
      end
    end

    it "raises NotImplementedError when dup is called" do
      model = Llama::Model.new(MODEL_PATH)
      context = model.context
      kv_cache = context.kv_cache
      expect_raises(NotImplementedError, "dup is not supported for Llama::KvCache") do
        kv_cache.dup
      end
    end
  end

  describe "basic operations" do
    it "can access KV cache from context" do
      model = Llama::Model.new(MODEL_PATH)
      context = model.context
      kv_cache = context.kv_cache

      kv_cache.should_not be_nil
      kv_cache.n_tokens.should eq(0)
      kv_cache.used_cells.should eq(0)
    end

    it "can clear the KV cache" do
      model = Llama::Model.new(MODEL_PATH)
      context = model.context
      kv_cache = context.kv_cache

      # Process a simple prompt to populate the cache
      prompt = "Hello, world!"
      tokens = model.vocab.tokenize(prompt)
      batch = Llama::Batch.get_one(tokens)
      context.decode(batch)

      # Cache should now have tokens
      kv_cache.n_tokens.should be > 0

      # Clear the cache
      kv_cache.clear

      # Cache should be empty again
      kv_cache.n_tokens.should eq(0)
    end
  end

  describe "method chaining" do
    it "supports basic method chaining for operations" do
      model = Llama::Model.new(MODEL_PATH)
      context = model.context
      kv_cache = context.kv_cache

      # Process a simple prompt to populate the cache
      prompt = "Hello, world!"
      tokens = model.vocab.tokenize(prompt)
      batch = Llama::Batch.get_one(tokens)
      context.decode(batch)

      # Verify cache has tokens
      kv_cache.n_tokens.should be > 0

      # Test method chaining - clear, defrag, and update in a single chain
      kv_cache.clear.defrag.update

      # Verify the operations were successful
      kv_cache.n_tokens.should eq(0)
    end

    it "supports method chaining for sequence operations" do
      model = Llama::Model.new(MODEL_PATH)
      context = model.context
      kv_cache = context.kv_cache

      # Process two prompts with different sequence IDs
      prompt1 = "Hello, world!"
      prompt2 = "How are you?"

      # First sequence (ID = 1)
      tokens1 = model.vocab.tokenize(prompt1)
      batch1 = Llama::Batch.new(tokens1.size)
      tokens1.each_with_index do |token, i|
        batch1.set_token(i, token, i, [1])
      end
      context.decode(batch1)

      # Second sequence (ID = 2)
      tokens2 = model.vocab.tokenize(prompt2)
      batch2 = Llama::Batch.new(tokens2.size)
      tokens2.each_with_index do |token, i|
        batch2.set_token(i, token, i, [2])
      end
      context.decode(batch2)

      # Test method chaining with sequence operations
      # Keep only sequence 1, add position delta, and update
      kv_cache.seq_keep(1).seq_add(1, 0, -1, 10).update

      # Verify operations were successful
      # Maximum position should be increased by the delta (10)
      max_pos = kv_cache.seq_pos_max(1)
      max_pos.should be >= tokens1.size - 1 + 10
    end
  end

  describe "sequence operations" do
    it "can get the maximum position in a sequence" do
      model = Llama::Model.new(MODEL_PATH)
      context = model.context
      kv_cache = context.kv_cache

      # Process a simple prompt to populate the cache
      prompt = "Hello, world!"
      tokens = model.vocab.tokenize(prompt)
      batch = Llama::Batch.get_one(tokens)
      context.decode(batch)

      # Get the maximum position (should be at least the number of tokens)
      max_pos = kv_cache.seq_pos_max(0)
      max_pos.should be >= tokens.size - 1
    end

    it "can check if the context supports KV cache shifting" do
      model = Llama::Model.new(MODEL_PATH)
      context = model.context
      kv_cache = context.kv_cache

      # This just tests that the method doesn't raise an error
      can_shift = kv_cache.can_shift?
      can_shift.should be_a(Bool)
    end

    it "can update the KV cache" do
      model = Llama::Model.new(MODEL_PATH)
      context = model.context
      kv_cache = context.kv_cache

      # Process a simple prompt to populate the cache
      prompt = "Hello, world!"
      tokens = model.vocab.tokenize(prompt)
      batch = Llama::Batch.get_one(tokens)
      context.decode(batch)

      # Update the cache
      kv_cache.update

      # This just tests that the method doesn't raise an error
    end
  end
end
