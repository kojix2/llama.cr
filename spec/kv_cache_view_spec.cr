require "./spec_helper"

# DEPRECATED: KV Cache View tests are temporarily disabled
# The KV cache view functionality has been removed from llama.cpp
# These tests will be re-enabled once the new memory management APIs are implemented

pending "Llama::KvCacheView (DEPRECATED - functionality removed from llama.cpp)" do
  describe "#initialization" do
    it "can create a KV cache view" do
      model = Llama::Model.new(MODEL_PATH)
      context = model.context
      view = Llama::KvCacheView.new(context)

      view.should_not be_nil
      view.n_cells.should be > 0
      view.n_seq_max.should eq(4) # Default value

      # Initial state should be empty
      view.token_count.should eq(0)
      view.used_cells.should eq(0)
      view.empty?.should be_true
      view.full?.should be_false

      # Free the view
      view.free
    end

    it "can create a KV cache view with custom n_seq_max" do
      model = Llama::Model.new(MODEL_PATH)
      context = model.context
      view = Llama::KvCacheView.new(context, n_seq_max: 8)

      view.n_seq_max.should eq(8)

      # Free the view
      view.free
    end

    it "supports block syntax for automatic resource management" do
      model = Llama::Model.new(MODEL_PATH)
      context = model.context

      # The view should be automatically freed after the block
      Llama::KvCacheView.new(context) do |view|
        view.should_not be_nil
        view.n_cells.should be > 0
      end
    end

    it "raises NotImplementedError when clone is called" do
      model = Llama::Model.new(MODEL_PATH)
      context = model.context
      view = Llama::KvCacheView.new(context)

      expect_raises(NotImplementedError, "clone is not supported for Llama::KvCacheView") do
        view.clone
      end

      view.free
    end

    it "raises NotImplementedError when dup is called" do
      model = Llama::Model.new(MODEL_PATH)
      context = model.context
      view = Llama::KvCacheView.new(context)

      expect_raises(NotImplementedError, "dup is not supported for Llama::KvCacheView") do
        view.dup
      end

      view.free
    end
  end

  describe "#operations" do
    it "can update the view" do
      model = Llama::Model.new(MODEL_PATH)
      context = model.context
      view = Llama::KvCacheView.new(context)

      # Process a simple prompt to populate the cache
      prompt = "Hello, world!"
      tokens = model.vocab.tokenize(prompt)
      batch = Llama::Batch.get_one(tokens)
      context.decode(batch)

      # Update the view
      view.update

      # Cache should now have tokens
      view.token_count.should be > 0
      view.used_cells.should be > 0
      view.empty?.should be_false

      view.free
    end

    it "supports method chaining" do
      model = Llama::Model.new(MODEL_PATH)
      context = model.context
      view = Llama::KvCacheView.new(context)

      # Process a simple prompt to populate the cache
      prompt = "Hello, world!"
      tokens = model.vocab.tokenize(prompt)
      batch = Llama::Batch.get_one(tokens)
      context.decode(batch)

      # Method chaining
      view.update.free
    end

    it "can access cells and sequences" do
      model = Llama::Model.new(MODEL_PATH)
      context = model.context
      view = Llama::KvCacheView.new(context)

      # Process a simple prompt to populate the cache
      prompt = "Hello, world!"
      tokens = model.vocab.tokenize(prompt)
      batch = Llama::Batch.get_one(tokens)
      context.decode(batch)

      # Update the view
      view.update

      # Access cells
      if view.used_cells > 0
        cell = view[0]
        cell.should be_a(Llama::KvCacheViewCell)
        cell.pos.should be >= 0

        # Access sequences
        sequences = view.sequences(0)
        sequences.should be_a(Array(Int32))
      end

      view.free
    end

    it "can iterate over cells" do
      model = Llama::Model.new(MODEL_PATH)
      context = model.context
      view = Llama::KvCacheView.new(context)

      # Process a simple prompt to populate the cache
      prompt = "Hello, world!"
      tokens = model.vocab.tokenize(prompt)
      batch = Llama::Batch.get_one(tokens)
      context.decode(batch)

      # Update the view
      view.update

      # Iterate over cells
      cell_count = 0
      view.each do |cell|
        cell.should be_a(Llama::KvCacheViewCell)
        cell_count += 1
      end

      cell_count.should eq(view.n_cells)

      view.free
    end

    it "provides aliases for common properties" do
      model = Llama::Model.new(MODEL_PATH)
      context = model.context
      view = Llama::KvCacheView.new(context)

      # Process a simple prompt to populate the cache
      prompt = "Hello, world!"
      tokens = model.vocab.tokenize(prompt)
      batch = Llama::Batch.get_one(tokens)
      context.decode(batch)

      # Update the view
      view.update

      # Check aliases
      view.size.should eq(view.token_count)
      view.capacity.should eq(view.n_cells)

      view.free
    end

    it "provides string representations" do
      model = Llama::Model.new(MODEL_PATH)
      context = model.context
      view = Llama::KvCacheView.new(context)

      # String representation
      view.to_s.should contain("KvCacheView")
      view.inspect.should contain("KvCacheView")

      # After freeing
      view.free
      view.to_s.should contain("freed")
    end
  end

  describe "#context_factory_method" do
    it "can create a KV cache view using context.kv_cache_view" do
      model = Llama::Model.new(MODEL_PATH)
      context = model.context
      view = context.kv_cache_view

      view.should_not be_nil
      view.n_cells.should be > 0
      view.n_seq_max.should eq(4) # Default value

      # Free the view
      view.free
    end

    it "supports block syntax with context.kv_cache_view" do
      model = Llama::Model.new(MODEL_PATH)
      context = model.context

      # The view should be automatically freed after the block
      context.kv_cache_view do |view|
        view.should_not be_nil
        view.n_cells.should be > 0
      end
    end
  end

  describe "#error_handling" do
    it "raises an error when accessing a freed view" do
      model = Llama::Model.new(MODEL_PATH)
      context = model.context
      view = Llama::KvCacheView.new(context)

      view.free

      expect_raises(Llama::KvCacheView::Error, "Attempt to use freed KV cache view") do
        view.n_cells
      end
    end

    it "raises an error when accessing an out-of-bounds index" do
      model = Llama::Model.new(MODEL_PATH)
      context = model.context
      view = Llama::KvCacheView.new(context)

      expect_raises(IndexError, "Index out of bounds") do
        view[view.n_cells]
      end

      view.free
    end
  end
end
