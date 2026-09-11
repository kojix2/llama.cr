require "./unit_helper"

describe Llama::GenerationOptions do
  it "validates values and defensively copies stop sequences" do
    expect_raises(ArgumentError, "max_tokens must be positive") { Llama::GenerationOptions.new(max_tokens: 0) }
    expect_raises(ArgumentError, "stop sequences must not be empty") { Llama::GenerationOptions.new(stop: [""]) }

    stops = ["END"]
    options = Llama::GenerationOptions.new(stop: stops)
    stops.clear
    options.stop.should eq(["END"])
    options.stop.clear
    options.stop.should eq(["END"])
  end
end

describe Llama::ContextOptions do
  it "validates batch and thread sizes" do
    expect_raises(ArgumentError, "batch_size must be positive") { Llama::ContextOptions.new(batch_size: 0_u32) }
    expect_raises(ArgumentError, "threads must be positive") { Llama::ContextOptions.new(threads: 0) }
  end
end

describe Llama::ModelOptions do
  it "retains typed model construction policy" do
    options = Llama::ModelOptions.new(gpu_layers: 3, vocab_only: true, check_tensors: true)
    options.gpu_layers.should eq(3)
    options.vocab_only.should be_true
    options.check_tensors.should be_true
  end
end
