require "./spec_helper"
require "../src/llama"

describe Llama::Sampler::Base do
  it "can create and free a sampler" do
    # Test creation and finalization of a sampler
    # This test just ensures that the sampler can be created and freed without errors
    sampler = Llama::Sampler::TopK.new(40)
    sampler.should_not be_nil
  end

  it "can create advanced samplers" do
    # Test creation of the new sampler types

    # Extended Temperature sampler
    temp_ext = Llama::Sampler::TempExt.new(0.8_f32, 0.5_f32, 1.0_f32)
    temp_ext.should_not be_nil

    # Top-N Sigma sampler
    top_n_sigma = Llama::Sampler::TopNSigma.new(2.0_f32)
    top_n_sigma.should_not be_nil

    # XTC sampler
    xtc = Llama::Sampler::Xtc.new(0.3_f32, 0.8_f32, 1)
    xtc.should_not be_nil

    # These tests require a vocabulary, so we'll skip them if we don't have a model
    model_path = ENV["MODEL_PATH"]? || ARGV.find { |arg| arg.starts_with?("--model=") }.try &.split("=")[1]?

    if model_path
      model = Llama::Model.new(model_path)
      vocab = model.vocab

      # Infill sampler
      infill = Llama::Sampler::Infill.new(vocab)
      infill.should_not be_nil

      # Grammar Lazy Patterns sampler
      grammar = %q{
        root ::= "test"
      }
      trigger_patterns = ["JSON:"]
      grammar_lazy = Llama::Sampler::GrammarLazyPatterns.new(
        vocab, grammar, "root", trigger_patterns
      )
      grammar_lazy.should_not be_nil
    else
      puts "Skipping tests that require a model"
    end
  end
end

describe Llama::SamplerChain do
  it "can create a sampler chain" do
    # Test creation of a sampler chain
    chain = Llama::SamplerChain.new
    chain.should_not be_nil
  end

  it "can add samplers to the chain" do
    # Test adding various samplers to the chain
    chain = Llama::SamplerChain.new
    chain.add(Llama::Sampler::TopK.new(40))
    chain.add(Llama::Sampler::TopP.new(0.95, 1))
    chain.add(Llama::Sampler::Temp.new(0.8))
    chain.add(Llama::Sampler::Dist.new)
    # If we get here without errors, the test passes
  end

  it "can sample tokens" do
    # Test sampling tokens from a context using the sampler chain
    # This test requires a model, so we'll skip it if the model file doesn't exist
    model_path = ENV["LLAMA_TEST_MODEL"]? || "spec/test_model.gguf"
    unless File.exists?(model_path)
      puts "Test model not available"
      next
    end

    model = Llama::Model.new(model_path)
    context = model.context

    chain = Llama::SamplerChain.new
    chain.add(Llama::Sampler::TopK.new(40))
    chain.add(Llama::Sampler::TopP.new(0.95, 1))
    chain.add(Llama::Sampler::Temp.new(0.8))
    chain.add(Llama::Sampler::Dist.new)

    # Process a simple prompt
    prompt = "Hello"
    input_tokens = model.vocab.tokenize(prompt)

    # Create a batch with the input tokens
    batch = Llama::LibLlama::LlamaBatch.new
    # Set up batch...

    # Process the batch
    context.decode(batch)

    # Sample a token
    token = chain.sample(context)
    token.should be_a(Int32)

    # Accept the token
    chain.accept(token)
    # If we get here without errors, the test passes
  end
end
