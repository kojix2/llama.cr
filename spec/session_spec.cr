require "./spec_helper"
require "file/tempfile"

describe Llama::Session do
  it "continues from prior visible text and resets" do
    model = Llama::Model.new(MODEL_PATH)
    options = Llama::GenerationOptions.new(max_tokens: 3, sampling: Llama::Sampling.greedy)
    session = model.session

    first = session.generate("Once upon a time", options)
    session.used_tokens.should be > first.usage.prompt_tokens
    second = session.generate(" and", options)

    fresh = model.context
    expected = fresh.complete("Once upon a time" + first.text + " and", options)
    second.text.should eq(expected.text)

    session.reset
    session.used_tokens.should eq(1) # tokenizer BOS for an empty transcript
    independent = session.generate(" and", options)
    fresh.complete(" and", options).text.should eq(independent.text)

    fresh.close
    session.close
    model.close
  end

  it "is closed with its model" do
    model = Llama::Model.new(MODEL_PATH)
    session = model.session
    model.close
    session.closed?.should be_true
    expect_raises(Llama::ClosedError) { session.used_tokens }
  end

  it "validates and restores portable snapshots before mutating state" do
    model = Llama::Model.new(MODEL_PATH)
    session = model.session
    options = Llama::GenerationOptions.new(max_tokens: 2, sampling: Llama::Sampling.greedy)
    session.generate("Once upon a time", options)
    snapshot = session.snapshot
    used_tokens = session.used_tokens

    incompatible = Llama::SessionSnapshot.new(snapshot.transcript, "different-model")
    expect_raises(Llama::StateCompatibilityError, "model fingerprint mismatch") do
      session.restore(incompatible)
    end
    session.used_tokens.should eq(used_tokens)

    file = File.tempfile("llama-session", ".json")
    begin
      session.save(file.path)
      session.reset
      session.load(file.path)
      session.used_tokens.should eq(used_tokens)
    ensure
      file.close
      file.delete
    end

    session.close
    model.close
  end
end
