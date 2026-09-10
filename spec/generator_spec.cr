require "./spec_helper"

describe Llama::Generator do
  it "returns typed usage and streams the same text" do
    model = Llama::Model.new(MODEL_PATH)
    context = model.context
    streamed = IO::Memory.new
    result = context.complete("Once upon a time", Llama::GenerationOptions.new(max_tokens: 5, sampling: Llama::Sampling.greedy)) do |chunk|
      chunk.text.valid_encoding?.should be_true
      streamed << chunk.text
    end

    streamed.to_s.should eq(result.text)
    result.usage.prompt_tokens.should be > 0
    result.usage.generated_tokens.should eq(result.sampled_tokens.size)
    context.close
    model.close
  end

  it "stops without emitting matched text" do
    model = Llama::Model.new(MODEL_PATH)
    context = model.context
    stages = [
      Llama::Sampling::Grammar.new(%(root ::= "abcENDtail")),
      Llama::Sampling::Greedy.new,
    ] of Llama::Sampling::Stage
    options = Llama::GenerationOptions.new(max_tokens: 16, stop: ["END"], sampling: Llama::Sampling::Plan.new(stages))
    result = context.complete("Story:", options)

    result.text.should eq("abc")
    result.finish_reason.should eq(Llama::FinishReason::StopSequence)
    result.stop_sequence.should eq("END")
    context.close
    model.close
  end

  it "returns cancellation before native prefill" do
    model = Llama::Model.new(MODEL_PATH)
    context = model.context
    cancellation = Llama::Cancellation.new
    cancellation.cancel
    result = context.complete("hello", Llama::GenerationOptions.new(cancellation: cancellation))
    result.finish_reason.should eq(Llama::FinishReason::Cancelled)
    result.text.should be_empty
    context.close
    model.close
  end
end
