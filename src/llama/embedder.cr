module Llama
  # Safe sentence embeddings over a dedicated embeddings-only context.
  class Embedder
    include NativeResource

    getter dimension : Int32

    def initialize(
      @model : Model,
      @pooling : Pooling = Pooling::Mean,
      @max_sequences : UInt32 = 8_u32,
      context_options : ContextOptions = ContextOptions.new,
    )
      raise ArgumentError.new("max_sequences must be positive") if @max_sequences == 0
      raise EmbeddingError.new("Pooling::None produces token vectors; use Context#get_embeddings_ith for token-level embeddings") if @pooling.none?
      raise EmbeddingError.new("model has neither an encoder nor a decoder") unless @model.has_encoder? || @model.has_decoder?

      @dimension = @model.n_embd_out
      raise EmbeddingError.new("model reports an invalid output embedding dimension: #{@dimension}") if @dimension <= 0

      @mutex = Mutex.new
      @running = false
      @context = Context.new(
        @model,
        n_ctx: context_options.context_size,
        n_batch: context_options.batch_size,
        n_threads: context_options.threads || 0,
        n_threads_batch: context_options.batch_threads || 0,
        embeddings: true,
        offload_kqv: context_options.offload_kqv,
        op_offload: context_options.op_offload,
        n_ubatch: context_options.micro_batch_size,
        n_seq_max: @max_sequences,
        pooling_type: @pooling.to_native
      )
    end

    def embed(text : String, normalize : Bool = false) : Array(Float32)
      embed_all([text], normalize).first
    end

    # Embeds inputs in native multi-sequence batches while preserving input order.
    def embed_all(texts : Enumerable(String), normalize : Bool = false) : Array(Array(Float32))
      inputs = texts.to_a
      return [] of Array(Float32) if inputs.empty?

      @context.with_operation do
        operation_started = false
        begin_operation!
        operation_started = true
        begin
          token_sets = inputs.map { |text| @model.vocab.tokenize(text) }
          token_sets.each_with_index do |tokens, index|
            raise EmbeddingError.new("input #{index} tokenized to an empty sequence") if tokens.empty?
            if tokens.size > @context.n_ctx_seq
              raise EmbeddingError.new("input #{index} exceeds sequence context size (#{tokens.size} tokens > #{@context.n_ctx_seq})")
            end
            if tokens.size > @context.n_batch
              # Pooling is performed for one native batch in this llama.cpp build;
              # splitting a sentence would silently change mean/CLS/last semantics.
              raise EmbeddingError.new("input #{index} exceeds embedding batch size (#{tokens.size} tokens > #{@context.n_batch})")
            end
          end

          vectors = [] of Array(Float32)
          groups(token_sets).each do |group|
            vectors.concat(embed_group(group))
          end

          normalize ? vectors.map { |vector| normalize_vector(vector) } : vectors
        ensure
          end_operation! if operation_started
        end
      end
    end

    def close : Nil
      @mutex.synchronize do
        return if closed?
        raise BusyError.new(self.class.to_s) if @running
        @context.close
      end
    end

    def free : Nil
      close
    end

    def closed? : Bool
      @context.closed?
    end

    def to_unsafe
      @context.to_unsafe
    end

    def finalize
      close
    rescue
    end

    private def groups(token_sets : Array(Array(Int32))) : Array(Array(Array(Int32)))
      result = [] of Array(Array(Int32))
      current = [] of Array(Int32)
      token_count = 0

      token_sets.each do |tokens|
        if !current.empty? && (current.size.to_u32 >= @max_sequences || token_count + tokens.size > @context.n_batch)
          result << current
          current = [] of Array(Int32)
          token_count = 0
        end
        current << tokens
        token_count += tokens.size
      end
      result << current unless current.empty?
      result
    end

    private def embed_group(token_sets : Array(Array(Int32))) : Array(Array(Float32))
      @context.memory.clear
      batch = Batch.new(token_sets.sum(&.size), 0, 1)
      batch_index = 0
      token_sets.each_with_index do |tokens, sequence_id|
        tokens.each_with_index do |token, position|
          batch.set_token(batch_index, token, position, [sequence_id], true)
          batch_index += 1
        end
      end

      if @model.has_encoder?
        @context.encode!(batch)
      else
        @context.decode!(batch)
      end

      token_sets.size.times.map do |sequence_id|
        @context.get_embeddings_seq(sequence_id) ||
          raise EmbeddingError.new("llama.cpp returned no embedding for sequence #{sequence_id}")
      end.to_a
    ensure
      batch.try(&.close)
    end

    private def normalize_vector(vector : Array(Float32)) : Array(Float32)
      squared_norm = vector.sum(0.0) { |value| value.to_f64 * value.to_f64 }
      raise EmbeddingError.new("cannot normalize a zero-norm embedding vector") unless squared_norm.finite? && squared_norm > 0.0

      inverse_norm = 1.0 / Math.sqrt(squared_norm)
      vector.map { |value| (value * inverse_norm).to_f32 }
    end

    private def begin_operation! : Nil
      @mutex.synchronize do
        ensure_open!
        raise BusyError.new(self.class.to_s) if @running
        @running = true
      end
    end

    private def end_operation! : Nil
      @mutex.synchronize { @running = false }
    end
  end
end
