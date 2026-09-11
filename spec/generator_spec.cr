require "./spec_helper"

class PromptBatchCancellation < Llama::Cancellation
  def initialize(@cancel_after_checks : Int32)
    super()
    @checks = 0
  end

  def cancelled? : Bool
    @checks += 1
    @checks >= @cancel_after_checks
  end
end

describe Llama::Generator do
  it "returns typed usage and streams the same text" do
    model = Llama::Model.new(MODEL_PATH)
    context = model.context
    streamed = IO::Memory.new
    emission_indices = [] of Int32
    result = context.complete("Once upon a time", Llama::GenerationOptions.new(max_tokens: 5, sampling: Llama::Sampling.greedy)) do |chunk|
      chunk.text.valid_encoding?.should be_true
      emission_indices << chunk.index
      streamed << chunk.text
    end

    streamed.to_s.should eq(result.text)
    emission_indices.should eq((0...emission_indices.size).to_a)
    result.usage.prompt_tokens.should be > 0
    result.usage.generated_tokens.should eq(result.sampled_tokens.size)
    context.close
    model.close
  end

  it "accepts typed model and context policy in one-shot completion" do
    result = Llama.complete(
      MODEL_PATH,
      "Once upon a time",
      Llama::GenerationOptions.new(max_tokens: 1, sampling: Llama::Sampling.greedy),
      model_options: Llama::ModelOptions.new(gpu_layers: 0),
      context_options: Llama::ContextOptions.new(context_size: 128_u32, batch_size: 32_u32)
    )
    result.usage.prompt_tokens.should be > 0
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

  it "includes streaming consumer time in end-to-end usage timing" do
    model = Llama::Model.new(MODEL_PATH)
    context = model.context
    plan = Llama::Sampling::Plan.new([
      Llama::Sampling::Grammar.new(%(root ::= "x")),
      Llama::Sampling::Greedy.new,
    ] of Llama::Sampling::Stage)
    delayed = false
    result = context.complete("Story:", Llama::GenerationOptions.new(max_tokens: 2, sampling: plan)) do |_chunk|
      unless delayed
        sleep 20.milliseconds
        delayed = true
      end
    end

    delayed.should be_true
    result.usage.generation_seconds.should be >= 0.02
    context.close
    model.close
  end

  it "keeps the context and model busy after rejecting a reentrant completion" do
    model = Llama::Model.new(MODEL_PATH)
    context = model.context
    plan = Llama::Sampling::Plan.new([
      Llama::Sampling::Grammar.new(%(root ::= "x")),
      Llama::Sampling::Greedy.new,
    ] of Llama::Sampling::Stage)
    checked = false

    context.complete("Story:", Llama::GenerationOptions.new(max_tokens: 2, sampling: plan)) do |_chunk|
      next if checked
      checked = true
      expect_raises(Llama::BusyError, "Llama::Context is busy") { context.complete("again") }
      expect_raises(Llama::BusyError, "Llama::Context is busy") { context.close }
      expect_raises(Llama::BusyError, "Llama::Model is busy") { model.close }
    end

    checked.should be_true
    context.close
    model.close
  end

  it "releases operation leases after a streaming callback exception" do
    model = Llama::Model.new(MODEL_PATH)
    context = model.context
    plan = Llama::Sampling::Plan.new([
      Llama::Sampling::Grammar.new(%(root ::= "x")),
      Llama::Sampling::Greedy.new,
    ] of Llama::Sampling::Stage)

    expect_raises(Exception, "callback boom") do
      context.complete("Story:", Llama::GenerationOptions.new(max_tokens: 2, sampling: plan)) do |_chunk|
        raise "callback boom"
      end
    end

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

  it "checks cancellation between prompt batches" do
    model = Llama::Model.new(MODEL_PATH)
    context = model.context(Llama::ContextOptions.new(batch_size: 2_u32, micro_batch_size: 2_u32))
    cancellation = PromptBatchCancellation.new(3)
    result = context.complete(
      "Once upon a time in a distant land",
      Llama::GenerationOptions.new(cancellation: cancellation)
    )

    result.finish_reason.should eq(Llama::FinishReason::Cancelled)
    result.sampled_tokens.should be_empty
    context.close
    model.close
  end
end
