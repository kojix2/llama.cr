module Llama
  # Stateful text facade. The canonical transcript is rebuilt before each turn;
  # this conservative strategy keeps stop-truncated text and native KV state in
  # agreement until incremental token reconciliation is introduced.
  class Session
    include NativeResource

    def initialize(@model : Model, context_options : ContextOptions = ContextOptions.new)
      @model_fingerprint = @model.fingerprint
      @context = @model.context(context_options)
      @transcript = ""
      @mutex = Mutex.new
      @running = false
    end

    def generate(prompt : String, options : GenerationOptions = GenerationOptions.new, &block : GenerationChunk ->) : Generation
      operation_started = false
      begin_operation!
      operation_started = true
      candidate = @transcript + prompt
      result = @context.complete(candidate, options, &block)
      if result.finish_reason.cancelled?
        @context.memory.clear
      else
        @transcript = candidate + result.text
      end
      result
    ensure
      end_operation! if operation_started
    end

    def generate(prompt : String, options : GenerationOptions = GenerationOptions.new) : Generation
      generate(prompt, options) { |_chunk| }
    end

    # Returns the historical tokenization count, including the vocabulary's
    # default special-token behavior. This is not native KV-cache occupancy.
    def used_tokens : Int32
      @mutex.synchronize do
        ensure_open!
        raise BusyError.new(self.class.to_s) if @running
        @model.vocab.tokenize(@transcript).size
      end
    end

    # Returns logical transcript tokens. An empty transcript is always zero;
    # callers can explicitly choose tokenizer special-token behavior.
    def transcript_token_count(add_special : Bool = true, parse_special : Bool = true) : Int32
      @mutex.synchronize do
        ensure_open!
        raise BusyError.new(self.class.to_s) if @running
        return 0 if @transcript.empty?
        @model.vocab.tokenize(@transcript, add_special, parse_special).size
      end
    end

    def reset : Nil
      @mutex.synchronize do
        ensure_open!
        raise BusyError.new(self.class.to_s) if @running
        @context.memory.clear
        @transcript = ""
      end
    end

    def snapshot : SessionSnapshot
      @mutex.synchronize do
        ensure_open!
        raise BusyError.new(self.class.to_s) if @running
        SessionSnapshot.new(@transcript, @model_fingerprint)
      end
    end

    # Restores canonical text only after every compatibility check succeeds.
    # The next generation rebuilds native KV state through checked decode.
    def restore(value : SessionSnapshot) : Nil
      @mutex.synchronize do
        ensure_open!
        raise BusyError.new(self.class.to_s) if @running
        validate_snapshot!(value)
        token_count = @model.vocab.tokenize(value.transcript).size
        if token_count > @context.n_ctx_seq
          raise StateCompatibilityError.new("session snapshot exceeds context size")
        end

        @context.memory.clear
        @transcript = value.transcript.dup
      end
    end

    def save(path : String) : Nil
      File.write(path, snapshot.to_json)
    end

    def load(path : String) : Nil
      if File.size(path) > SessionSnapshot::MAX_SERIALIZED_BYTES
        raise StateCompatibilityError.new("session snapshot is too large")
      end
      restore(SessionSnapshot.from_json(File.read(path)))
    rescue ex : JSON::ParseException
      raise StateCompatibilityError.new("invalid session snapshot JSON")
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

    private def begin_operation! : Nil
      acquired = false
      @mutex.synchronize do
        ensure_open!
        raise BusyError.new(self.class.to_s) if @running
        @running = true
        acquired = true
      end
      @model.begin_operation!
    rescue ex
      @mutex.synchronize { @running = false } if acquired
      raise ex
    end

    private def end_operation! : Nil
      @model.end_operation!
      @mutex.synchronize { @running = false }
    end

    private def validate_snapshot!(value : SessionSnapshot) : Nil
      unless value.format_version == SessionSnapshot::FORMAT_VERSION
        raise StateCompatibilityError.new("unsupported session snapshot format #{value.format_version}")
      end
      unless value.llama_cpp_version == Llama.llama_cpp_version
        raise StateCompatibilityError.new("session snapshot llama.cpp version mismatch")
      end
      unless value.model_fingerprint == @model_fingerprint
        raise StateCompatibilityError.new("session snapshot model fingerprint mismatch")
      end
    end
  end
end
