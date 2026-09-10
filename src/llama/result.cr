module Llama
  alias Token = Int32

  enum FinishReason
    EndOfGeneration
    StopSequence
    Length
    ContextFull
    Cancelled
  end

  record GenerationChunk,
    text : String,
    token : Token,
    index : Int32

  record Usage,
    prompt_tokens : Int32,
    generated_tokens : Int32,
    prompt_seconds : Float64,
    generation_seconds : Float64 do
    def tokens_per_second : Float64
      return 0.0 if generation_seconds <= 0.0
      generated_tokens / generation_seconds
    end
  end

  # Immutable summary of one generation operation.
  struct Generation
    getter text : String
    getter finish_reason : FinishReason
    getter stop_sequence : String?
    getter usage : Usage

    def initialize(
      @text : String,
      sampled_tokens : Array(Token),
      @finish_reason : FinishReason,
      @stop_sequence : String?,
      @usage : Usage,
    )
      @sampled_tokens = sampled_tokens.dup
    end

    # Returns a defensive copy so result values cannot expose internal ledgers.
    def sampled_tokens : Array(Token)
      @sampled_tokens.dup
    end

    @sampled_tokens : Array(Token)
  end
end
