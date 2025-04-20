require "./spec_helper"

describe Llama do
  describe ".system_info" do
    it "returns system information as a string" do
      info = Llama.system_info
      info.should be_a(String)
      info.should_not be_empty
      puts "  - System info: #{info.lines.first}"
    end
  end

  describe "time API" do
    it "returns increasing microseconds and milliseconds" do
      t0_us = Llama.time_us
      t0_ms = Llama.time_ms
      sleep 0.01
      t1_us = Llama.time_us
      t1_ms = Llama.time_ms
      (t1_us > t0_us).should be_true
      (t1_ms > t0_ms).should be_true
      ((t1_us - t0_us) / 1000).should be_close(t1_ms - t0_ms, 2)
    end

    it "measures elapsed time in ms for a block" do
      elapsed = Llama.measure_ms { sleep 0.02 }
      elapsed.should be > 0.0
      # Do not check the exact value due to possible environment delays
    end
  end

  describe "Error handling" do
    it "raises an error when loading a non-existent model" do
      expect_raises(Llama::Model::Error, /Failed to load model/) do
        Llama::Model.new("non_existent_model.gguf")
      end
    end

    it "raises an error with invalid parameters" do
      expect_raises(ArgumentError) do
        # Negative max_tokens should raise an error
        Llama.generate("dummy_path.gguf", "test", max_tokens: -10)
      end
    end
  end

  # Model-dependent tests
  # These tests use the MODEL_PATH environment variable or --model= command line argument
  model_path = ENV["MODEL_PATH"]? || ARGV.find { |arg| arg.starts_with?("--model=") }.try &.split("=")[1]?

  if model_path.nil?
    pending "Skipping model-dependent tests (no model provided)"
  else
    describe ".generate" do
      it "generates text from a prompt" do
        prompt = "Once upon a time"
        response = Llama.generate(model_path, prompt, max_tokens: 10)
        response.should be_a(String)
        response.should_not be_empty
        puts "  - Generated: '#{response}'"
      end

      it "handles empty prompts" do
        response = Llama.generate(model_path, "", max_tokens: 5)
        response.should be_a(String)
        puts "  - Generated from empty prompt: '#{response}'"
      end

      it "handles special characters in prompts" do
        prompt = "Hello! 你好! こんにちは! 안녕하세요!"
        response = Llama.generate(model_path, prompt, max_tokens: 5)
        response.should be_a(String)
        puts "  - Prompt with special chars: '#{prompt}'"
        puts "  - Generated: '#{response}'"
      end
    end
  end
end
