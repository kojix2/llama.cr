require "./spec_helper"

describe Llama::KvCache do
  # KvCache tests require a model
  model_path = ENV["MODEL_PATH"]? || ARGV.find { |arg| arg.starts_with?("--model=") }.try &.split("=")[1]?

  if model_path.nil?
    pending "Skipping KvCache tests (no model provided)"
  else
    describe "basic operations" do
      it "can access KV cache from context" do
        model = Llama::Model.new(model_path)
        context = model.context
        kv_cache = context.kv_cache

        kv_cache.should_not be_nil
        kv_cache.n_tokens.should eq(0)
        kv_cache.used_cells.should eq(0)
      end

      it "can clear the KV cache" do
        model = Llama::Model.new(model_path)
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

    describe "sequence operations" do
      it "can get the maximum position in a sequence" do
        model = Llama::Model.new(model_path)
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
        model = Llama::Model.new(model_path)
        context = model.context
        kv_cache = context.kv_cache

        # This just tests that the method doesn't raise an error
        can_shift = kv_cache.can_shift?
        can_shift.should be_a(Bool)
      end

      it "can update the KV cache" do
        model = Llama::Model.new(model_path)
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
end
