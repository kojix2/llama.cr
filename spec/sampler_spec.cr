require "./spec_helper"
require "../src/llama"

describe Llama::Sampler::Base do
  it "can create and free a sampler" do
    # Test creation and finalization of a sampler
    # This test just ensures that the sampler can be created and freed without errors
    sampler = Llama::Sampler::TopK.new(40)
    sampler.should_not be_nil
  end

  it "raises NotImplementedError when clone is called" do
    sampler = Llama::Sampler::TopK.new(40)
    expect_raises(NotImplementedError, "clone is not supported for Llama::Sampler::TopK") do
      sampler.clone
    end
  end

  it "raises NotImplementedError when dup is called" do
    sampler = Llama::Sampler::TopK.new(40)
    expect_raises(NotImplementedError, "dup is not supported for Llama::Sampler::TopK") do
      sampler.dup
    end
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

    model = Llama::Model.new(MODEL_PATH)
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
  end
end

describe Llama::SamplerChain do
  it "can create a sampler chain" do
    # Test creation of a sampler chain
    chain = Llama::SamplerChain.new
    chain.should_not be_nil
  end

  it "raises NotImplementedError when clone is called" do
    chain = Llama::SamplerChain.new
    expect_raises(NotImplementedError, "clone is not supported for Llama::SamplerChain") do
      chain.clone
    end
  end

  it "raises NotImplementedError when dup is called" do
    chain = Llama::SamplerChain.new
    expect_raises(NotImplementedError, "dup is not supported for Llama::SamplerChain") do
      chain.dup
    end
  end

  it "can add samplers to the chain" do
    # Test adding various samplers to the chain
    chain = Llama::SamplerChain.new
    k = Llama::Sampler::TopK.new(40)
    p = Llama::Sampler::TopP.new(0.95, 1)
    t = Llama::Sampler::Temp.new(0.8)
    d = Llama::Sampler::Dist.new
    chain.add(k)
    chain.add(p)
    chain.add(t)
    chain.add(d)
    # If we get here without errors, the test passes
  end

  it "can remove samplers from the chain and restore ownership" do
    chain = Llama::SamplerChain.new
    k = Llama::Sampler::TopK.new(40)
    p = Llama::Sampler::TopP.new(0.95, 1)
    t = Llama::Sampler::Temp.new(0.8)
    d = Llama::Sampler::Dist.new
    chain.add(k)
    chain.add(p)
    chain.add(t)
    chain.add(d)
    # Remove the second sampler (TopP)
    removed = chain.remove(1)
    removed.should be(p)
    # Ownership should be restored, so it can be added to another chain
    chain2 = Llama::SamplerChain.new
    chain2.add(removed)
    # Remove again from the new chain
    removed2 = chain2.remove(0)
    removed2.should be(removed)
    # If we get here without errors, the test passes
  end

  it "can sample tokens" do
    # Test sampling tokens from a context using the sampler chain

    model = Llama::Model.new(MODEL_PATH)
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
    batch = Llama::Batch.from_tokens(input_tokens, true)

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
