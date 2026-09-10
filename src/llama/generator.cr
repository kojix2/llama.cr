module Llama
  # Shared checked generation loop used by typed high-level APIs.
  class Generator
    def initialize(@context : Context, @options : GenerationOptions)
      @model = @context.generator_model
      @tokenizer = Tokenizer.new(@model.vocab)
    end

    def generate(prompt : String, &block : GenerationChunk ->) : Generation
      if @model.has_encoder?
        raise UnsupportedOperationError.new("encoder-decoder generation is not supported yet")
      end

      prompt_tokens = @tokenizer.encode(prompt)
      raise TokenizationError.new("Tokenization resulted in empty token array") if prompt_tokens.empty?
      if prompt_tokens.size > @context.n_ctx
        raise ContextFullError.new("Prompt exceeds context size [tokens: #{prompt_tokens.size}, n_ctx: #{@context.n_ctx}]")
      end

      prompt_started = Time.instant
      if cancelled?
        return result("", Array(Token).new, FinishReason::Cancelled, nil, prompt_tokens.size, 0.0, 0.0)
      end
      @context.generator_prefill(prompt_tokens)
      prompt_seconds = (Time.instant - prompt_started).total_seconds

      chain = @options.sampling.build(@model.vocab)
      decoder = StreamingDecoder.new(@tokenizer)
      detector = StopDetector.new(@options.stop, @options.include_stop)
      sampled = [] of Token
      output = IO::Memory.new
      finish_reason = FinishReason::Length
      position = prompt_tokens.size
      generation_started = Time.instant

      @options.max_tokens.times do |index|
        if cancelled?
          finish_reason = FinishReason::Cancelled
          break
        end

        token = chain.sample(@context)
        if @model.vocab.eog?(token)
          finish_reason = FinishReason::EndOfGeneration
          break
        end
        sampled << token

        # Match llama_detokenize's historical handling of the first generated
        # piece while retaining byte-safe streaming for subsequent pieces.
        piece = @tokenizer.piece(token, sampled.size == 1 ? 1 : 0)
        emit(detector.push(decoder.push(piece)), token, index, output, &block)
        if detector.matched
          finish_reason = FinishReason::StopSequence
          break
        end

        if position >= @context.n_ctx
          finish_reason = FinishReason::ContextFull
          break
        end
        break if index == @options.max_tokens - 1

        @context.generator_decode(token, position)
        position += 1
      end

      tail = detector.push(decoder.finish)
      emit(tail, sampled.last? || TOKEN_NULL, sampled.size, output, &block)
      finish_reason = FinishReason::StopSequence if detector.matched
      emit(detector.finish, sampled.last? || TOKEN_NULL, sampled.size, output, &block)
      generation_seconds = (Time.instant - generation_started).total_seconds

      result(output.to_s, sampled, finish_reason, detector.matched, prompt_tokens.size, prompt_seconds, generation_seconds)
    ensure
      chain.try(&.close)
    end

    private def emit(text : String, token : Token, index : Int32, output : IO::Memory, &block : GenerationChunk ->) : Nil
      return if text.empty?
      output << text
      yield GenerationChunk.new(text, token, index)
    end

    private def cancelled? : Bool
      @options.cancellation.try(&.cancelled?) || false
    end

    private def result(text, tokens, reason, stop, prompt_tokens, prompt_seconds, generation_seconds) : Generation
      Generation.new(text, tokens, reason, stop, Usage.new(prompt_tokens, tokens.size, prompt_seconds, generation_seconds))
    end
  end
end
