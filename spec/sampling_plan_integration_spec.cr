require "./integration_helper"

describe Llama::Sampling::Plan do
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

  it "cleans up a partially built chain when a later stage fails" do
    model = Llama::Model.new(MODEL_PATH)
    plan = Llama::Sampling::Plan.new([
      Llama::Sampling::TopK.new(10),
      Llama::Sampling::Grammar.new("this is not valid GBNF"),
      Llama::Sampling::Greedy.new,
    ] of Llama::Sampling::Stage)

    expect_raises(Llama::Error, "Failed to create Grammar sampler") do
      plan.build(model.vocab)
    end

    # A failed build must not poison later sampler construction.
    Llama::Sampling.greedy.build(model.vocab).close
    model.close
  end
end
