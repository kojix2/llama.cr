require "./spec_helper"

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
    expect_raises(ArgumentError, "sequence_count must be positive") { Llama::ContextOptions.new(sequence_count: 0_u32) }
  end
end

describe Llama::EmbeddingOptions do
  it "validates the native sequence capacity" do
    expect_raises(ArgumentError, "max_sequences must be positive") do
      Llama::EmbeddingOptions.new(max_sequences: 0_u32)
    end
  end
end

describe Llama::ModelOptions do
  it "loads models through typed options" do
    options = Llama::ModelOptions.new(vocab_only: true, check_tensors: true)
    Llama::Model.open(MODEL_PATH, options) do |model|
      model.n_params.should be > 0
    end
  end
end
