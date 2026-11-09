require "spec"
require "../src/llama/raw"

describe "require llama/raw" do
  it "exposes the pinned upstream declarations without the wrapper API" do
    Llama::LibLlama::LLAMA_DEFAULT_SEED.should eq(0xffff_ffff_u32)
    Llama::LibLlama::LLAMA_TOKEN_NULL.should eq(-1)
    Llama::LibLlama::LLAMA_SESSION_VERSION.should eq(11)
    Llama::LibLlama::LLAMA_STATE_SEQ_VERSION.should eq(4)
    Llama::LibLlama::LLAMA_DRY_MAX_SEQ_BREAKERS.should eq(64)
    Llama::LibLlama::LlamaVocabType::TEST.value.should eq(7)
    Llama::LibLlama::LlamaVocabType::PLAMO3.value.should eq(8)
    Llama::LibLlama::LlamaFtype::MOSTLY_F8_E4M3.value.should eq(42)

    token = 7.as(Llama::LibLlama::LlamaToken)
    token.should eq(7)
  end
end
