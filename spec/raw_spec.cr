require "spec"
require "../src/llama/raw"

describe "require llama/raw" do
  it "exposes the pinned upstream declarations without the wrapper API" do
    Llama::LibLlama::LLAMA_DEFAULT_SEED.should eq(0xffff_ffff_u32)
    Llama::LibLlama::LLAMA_TOKEN_NULL.should eq(-1)

    token = 7.as(Llama::LibLlama::LlamaToken)
    token.should eq(7)
  end
end
