require "./spec_helper"

describe Llama::Cancellation do
  it "cancels idempotently" do
    cancellation = Llama::Cancellation.new
    cancellation.cancelled?.should be_false
    cancellation.cancel
    cancellation.cancel
    cancellation.cancelled?.should be_true
  end
end

describe Llama::Usage do
  it "calculates generation throughput" do
    Llama::Usage.new(4, 10, 0.1, 2.0).tokens_per_second.should eq(5.0)
    Llama::Usage.new(4, 10, 0.1, 0.0).tokens_per_second.should eq(0.0)
  end
end

describe Llama::Generation do
  it "does not alias sampled token arrays" do
    tokens = [1, 2]
    generation = Llama::Generation.new(
      "text",
      tokens,
      Llama::FinishReason::Length,
      nil,
      Llama::Usage.new(1, 2, 0.0, 1.0)
    )

    tokens << 3
    returned = generation.sampled_tokens
    returned << 4
    generation.sampled_tokens.should eq([1, 2])
  end
end
