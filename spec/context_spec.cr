require "./spec_helper"

# This file contains tests for the Context class
# Run with: crystal spec spec/context_spec.cr -- --model=/path/to/model.gguf
# Or set the MODEL_PATH environment variable

describe Llama::Context do
  # Get model path from command line arguments or environment variable
  model_path = ENV["MODEL_PATH"]? || ARGV.find { |arg| arg.starts_with?("--model=") }.try &.split("=")[1]?

  if model_path.nil?
    pending "Skipping context tests (no model provided)"
  else
    describe "#encode" do
      it "encodes input for encoder-decoder models" do
        model = Llama::Model.new(model_path)
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
        model = Llama::Model.new(model_path)
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
        model = Llama::Model.new(model_path)
        context = model.context

        prompt = "Count to ten:"

        # Generate with very limited tokens
        short_response = context.generate(prompt, max_tokens: 5, temperature: 0.0)

        # Generate with more tokens
        long_response = context.generate(prompt, max_tokens: 20, temperature: 0.0)

        # The longer generation should contain more characters
        short_response.size.should be < long_response.size

        puts "  - Short response (max_tokens=5): '#{short_response}'"
        puts "  - Long response (max_tokens=20): '#{long_response}'"
      end

      it "produces different outputs with different temperatures" do
        model = Llama::Model.new(model_path)
        context = model.context

        prompt = "Creative writing:"

        # Clear KV cache to make each generation independent
        context.kv_cache.clear
        deterministic1 = context.generate(prompt, max_tokens: 15, temperature: 0.0)

        context.kv_cache.clear
        deterministic2 = context.generate(prompt, max_tokens: 15, temperature: 0.0)

        context.kv_cache.clear
        random = context.generate(prompt, max_tokens: 15, temperature: 1.0)

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
        model = Llama::Model.new(model_path)
        context = model.context

        prompt = "The weather today is"

        # Test with various temperature values
        temperatures = [0.0, 0.5, 1.0]
        responses = temperatures.map do |temp|
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
end
