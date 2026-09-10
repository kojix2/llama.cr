require "./spec_helper"

describe Llama::Sampling::Plan do
  it "validates selector placement" do
    expect_raises(ArgumentError, "sampling plan cannot be empty") do
      Llama::Sampling::Plan.new([] of Llama::Sampling::Stage)
    end
    expect_raises(ArgumentError, "sampling plan must end with a selector") do
      Llama::Sampling::Plan.new([Llama::Sampling::TopK.new(10)] of Llama::Sampling::Stage)
    end
  end

  it "defensively copies stage arrays" do
    stages = [Llama::Sampling::Greedy.new] of Llama::Sampling::Stage
    plan = Llama::Sampling::Plan.new(stages)
    stages.clear
    plan.stages.size.should eq(1)
    plan.stages.clear
    plan.stages.size.should eq(1)
  end

  it "builds independent native chains" do
    model = Llama::Model.new(MODEL_PATH)
    plan = Llama::Sampling.greedy
    first = plan.build(model.vocab)
    second = plan.build(model.vocab)
    first.to_unsafe.should_not eq(second.to_unsafe)
    first.close
    second.close
    model.close
  end
end
