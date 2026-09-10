require "./spec_helper"

describe Llama::DecodeError do
  it "classifies all documented llama_decode statuses" do
    Llama::DecodeError.reason_for(1).should eq(Llama::DecodeError::Reason::NoKvSlot)
    Llama::DecodeError.reason_for(2).should eq(Llama::DecodeError::Reason::Aborted)
    Llama::DecodeError.reason_for(-1).should eq(Llama::DecodeError::Reason::InvalidBatch)
    Llama::DecodeError.reason_for(-2).should eq(Llama::DecodeError::Reason::Fatal)
  end

  it "retains native decode diagnostics" do
    error = Llama::DecodeError.new(
      "llama_decode",
      1,
      Llama::DecodeError::Reason::NoKvSlot,
      32
    )

    error.operation.should eq("llama_decode")
    error.native_code.should eq(1)
    error.reason.should eq(Llama::DecodeError::Reason::NoKvSlot)
    error.batch_size.should eq(32)
  end
end
