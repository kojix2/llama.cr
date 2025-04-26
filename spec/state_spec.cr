require "./spec_helper"
require "file_utils"

describe Llama::State do
  describe "#clone_dup" do
    it "raises NotImplementedError when clone is called" do
      model = Llama::Model.new(MODEL_PATH)
      context = model.context
      state = context.state
      expect_raises(NotImplementedError, "clone is not supported for Llama::State") do
        state.clone
      end
    end

    it "raises NotImplementedError when dup is called" do
      model = Llama::Model.new(MODEL_PATH)
      context = model.context
      state = context.state
      expect_raises(NotImplementedError, "dup is not supported for Llama::State") do
        state.dup
      end
    end
  end

  describe "basic operations" do
    it "can access state from context" do
      model = Llama::Model.new(MODEL_PATH)
      context = model.context
      state = context.state

      state.should_not be_nil
    end

    it "can get state size" do
      model = Llama::Model.new(MODEL_PATH)
      context = model.context
      state = context.state

      # Process a simple prompt to populate the state
      prompt = "Hello world!"
      tokens = model.vocab.tokenize(prompt)
      batch = Llama::Batch.get_one(tokens)
      context.decode(batch)

      # Get the state size
      size = state.size
      size.should be > 0
    end

    it "can get and set state data" do
      model = Llama::Model.new(MODEL_PATH)
      context = model.context
      state = context.state

      # Process a simple prompt to populate the state
      prompt = "Hello world!"
      tokens = model.vocab.tokenize(prompt)
      batch = Llama::Batch.get_one(tokens)
      context.decode(batch)

      # Get the state data
      data = state.get_data
      data.should_not be_empty

      # Create a new context
      new_context = model.context
      new_state = new_context.state

      # Set the state data
      bytes_read = new_state.set_data(data)
      bytes_read.should eq(data.size)
    end
  end

  describe "file operations" do
    # Temporary file for testing
    temp_file = "spec_state_test.bin"

    # Clean up after tests
    after_each do
      File.delete(temp_file) if File.exists?(temp_file)
    end

    it "can save and load state to/from a file" do
      model = Llama::Model.new(MODEL_PATH)
      context = model.context
      state = context.state

      # Process a simple prompt to populate the state
      prompt = "Hello world!"
      tokens = model.vocab.tokenize(prompt)
      batch = Llama::Batch.get_one(tokens)
      context.decode(batch)

      # Save the state to a file
      success = state.save_file(temp_file, tokens)
      success.should be_true
      File.exists?(temp_file).should be_true

      # Create a new context
      new_context = model.context
      new_state = new_context.state

      # Load the state from the file
      loaded_tokens = new_state.load_file(temp_file)
      loaded_tokens.should eq(tokens)
    end

    it "raises an error when loading from a non-existent file" do
      model = Llama::Model.new(MODEL_PATH)
      context = model.context
      state = context.state

      expect_raises(Llama::Error) do
        state.load_file("non_existent_file.bin")
      end
    end
  end

  describe "sequence operations" do
    # Temporary file for testing
    temp_file = "spec_seq_state_test.bin"

    # Clean up after tests
    after_each do
      File.delete(temp_file) if File.exists?(temp_file)
    end

    it "can get sequence state size" do
      model = Llama::Model.new(MODEL_PATH)
      context = model.context
      state = context.state

      # Process a simple prompt to populate the state
      prompt = "Hello world!"
      tokens = model.vocab.tokenize(prompt)
      batch = Llama::Batch.get_one(tokens)
      context.decode(batch)

      # Get the sequence state size
      size = state.seq_size(0)
      size.should be > 0
    end

    it "can get and set sequence state data" do
      model = Llama::Model.new(MODEL_PATH)
      context = model.context
      state = context.state

      # Process a simple prompt to populate the state
      prompt = "Hello world!"
      tokens = model.vocab.tokenize(prompt)
      batch = Llama::Batch.get_one(tokens)
      context.decode(batch)

      # Get the sequence state data
      data = state.seq_get_data(0)
      data.should_not be_empty

      # Create a new context
      new_context = model.context
      new_state = new_context.state

      # Set the sequence state data
      bytes_read = new_state.seq_set_data(data, 0)
      bytes_read.should be > 0
    end

    it "can save and load sequence state to/from a file" do
      model = Llama::Model.new(MODEL_PATH)
      context = model.context
      state = context.state

      # Process a simple prompt to populate the state
      prompt = "Hello world!"
      tokens = model.vocab.tokenize(prompt)
      batch = Llama::Batch.get_one(tokens)
      context.decode(batch)

      # Save the sequence state to a file
      bytes_written = state.seq_save_file(temp_file, 0, tokens)
      bytes_written.should be > 0
      File.exists?(temp_file).should be_true

      # Create a new context
      new_context = model.context
      new_state = new_context.state

      # Load the sequence state from the file
      loaded_tokens = new_state.seq_load_file(temp_file, 0)
      loaded_tokens.should eq(tokens)
    end
  end
end
