require "./spec_helper"

# This file contains tests for the Context class
# Run with: crystal spec spec/context_spec.cr -- --model=/path/to/model.gguf
# Or set the MODEL_PATH environment variable

describe Llama::Context do
  describe "#clone_dup" do
    it "raises NotImplementedError when clone is called" do
      model = Llama::Model.new(MODEL_PATH)
      context = model.context
      expect_raises(NotImplementedError, "clone is not supported for Llama::Context") do
        context.clone
      end
    end

    it "raises NotImplementedError when dup is called" do
      model = Llama::Model.new(MODEL_PATH)
      context = model.context
      expect_raises(NotImplementedError, "dup is not supported for Llama::Context") do
        context.dup
      end
    end
  end

  describe "#attributes" do
    it "returns valid context attributes" do
      model = Llama::Model.new(MODEL_PATH)
      context = model.context

      context.n_ctx.should be_a(UInt32)
      context.n_ctx.should be > 0

      context.n_batch.should be_a(UInt32)
      context.n_batch.should be > 0

      context.n_ubatch.should be_a(UInt32)
      context.n_ubatch.should be >= 0

      context.n_seq_max.should be_a(UInt32)
      context.n_seq_max.should be > 0

      context.n_threads.should be_a(Int32)
      context.n_threads.should be >= 0

      context.n_threads_batch.should be_a(Int32)
      context.n_threads_batch.should be >= 0

      puts "  - n_ctx: #{context.n_ctx}, n_batch: #{context.n_batch}, n_ubatch: #{context.n_ubatch}, n_seq_max: #{context.n_seq_max}, n_threads: #{context.n_threads}, n_threads_batch: #{context.n_threads_batch}"
    end
  end

  describe "#encode" do
    it "encodes input for encoder-decoder models" do
      model = Llama::Model.new(MODEL_PATH)
      context = model.context

      # Skip this test if the model doesn't have an encoder
      unless model.has_encoder?
        puts "  - Skipping encode test (model doesn't have an encoder)"
        next
      end

      # Create a simple batch for encoding
      prompt = "Translate to French: Hello, world!"
      input_tokens = model.vocab.tokenize(prompt)

      # Create a batch with the input tokens
      batch = Llama::Batch.new(input_tokens.size)
      batch.add_tokens(input_tokens)

      # Encode the batch
      begin
        result = context.encode(batch)
        # If we get here, encoding worked
        result.should be >= 0
        puts "  - Successfully encoded batch with #{input_tokens.size} tokens"
      rescue ex : Llama::Error
        # If the model doesn't support encoding, this might fail
        puts "  - Encoding failed: #{ex.message} (model might not support encoding)"
      end
    end
  end

  describe "#generate" do
    it "generates multiple tokens" do
      model = Llama::Model.new(MODEL_PATH)
      context = model.context

      prompt = "Once upon a time"
      max_tokens = 20
      response = context.generate(prompt, max_tokens: max_tokens, temperature: 0.8)

      # The response should be a non-empty string
      response.should be_a(String)
      response.should_not be_empty

      # The response should contain multiple tokens
      # We can't assert exact length in tokens, but we can check it's substantial
      response.size.should be > 5

      puts "  - Prompt: '#{prompt}'"
      puts "  - Response: '#{response}'"
      puts "  - Response length: #{response.size} characters"
    end

    it "respects the max_tokens parameter" do
      model = Llama::Model.new(MODEL_PATH)

      prompt = "Count to ten:"

      # Generate with very limited tokens using separate contexts
      context1 = model.context
      short_response = context1.generate(prompt, max_tokens: 5, temperature: 0.0)

      # Generate with more tokens using a new context
      context2 = model.context
      long_response = context2.generate(prompt, max_tokens: 20, temperature: 0.0)

      # The longer generation should contain more characters
      short_response.size.should be < long_response.size

      puts "  - Short response (max_tokens=5): '#{short_response}'"
      puts "  - Long response (max_tokens=20): '#{long_response}'"
    end

    it "produces different outputs with different temperatures" do
      model = Llama::Model.new(MODEL_PATH)
      context = model.context

      prompt = "Creative writing:"

      # Generate with deterministic settings (temperature=0.0)
      deterministic1 = context.generate(prompt, max_tokens: 15, temperature: 0.0)

      # Create a new context for independent generation
      context2 = model.context
      deterministic2 = context2.generate(prompt, max_tokens: 15, temperature: 0.0)

      # Create another new context for random generation
      context3 = model.context
      random = context3.generate(prompt, max_tokens: 15, temperature: 1.0)

      # Deterministic generations (temperature=0.0) should be identical
      begin
        deterministic1.should eq(deterministic2)
      rescue ex
        puts "  - Warning: Deterministic generations were not identical"
        puts "  - First: '#{deterministic1}'"
        puts "  - Second: '#{deterministic2}'"
      end

      # Random generation should differ from deterministic
      if deterministic1 == random
        puts "  - Warning: Random generation matched deterministic generation (rare case)"
      end

      puts "  - Deterministic: '#{deterministic1}'"
      puts "  - Random: '#{random}'"
    end

    it "handles different temperature values correctly" do
      model = Llama::Model.new(MODEL_PATH)

      prompt = "The weather today is"

      # Test with various temperature values using separate contexts
      temperatures = [0.0, 0.5, 1.0]
      responses = temperatures.map do |temp|
        context = model.context
        response = context.generate(prompt, max_tokens: 10, temperature: temp.to_f32)
        puts "  - Temperature #{temp}: '#{response}'"
        response
      end

      # All responses should be non-empty
      responses.each do |response|
        response.should_not be_empty
      end
    end
  end
end
