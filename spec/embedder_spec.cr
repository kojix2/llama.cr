require "./spec_helper"

describe Llama::Embedder do
  it "copies normalized vectors and preserves batch order" do
    model = Llama::Model.new(MODEL_PATH)
    embedder = model.embedder

    first = embedder.embed("hello", normalize: true)
    batch = embedder.embed_all(["hello", "world"], normalize: true)

    first.size.should eq(model.n_embd_out)
    batch.map(&.size).should eq([model.n_embd_out, model.n_embd_out])
    Math.sqrt(first.sum(0.0) { |value| value.to_f64 * value.to_f64 }).should be_close(1.0, 1e-5)
    first.zip(batch[0]).sum(0.0) { |pair| (pair[0] - pair[1]).abs.to_f64 }.should be_close(0.0, 1e-5)

    # A later native call must not mutate an already returned Crystal array.
    snapshot = first.dup
    embedder.embed("later")
    first.should eq(snapshot)

    embedder.close
    model.close
  end

  it "defines empty batch and token-level pooling behavior explicitly" do
    model = Llama::Model.new(MODEL_PATH)
    embedder = model.embedder
    embedder.embed_all([] of String).should be_empty
    embedder.close

    expect_raises(Llama::EmbeddingError, "Pooling::None produces token vectors") do
      model.embedder(pooling: Llama::Pooling::None)
    end
    expect_raises(ArgumentError, "max_sequences must be positive") do
      model.embedder(max_sequences: 0_u32)
    end
    model.close
  end
end
