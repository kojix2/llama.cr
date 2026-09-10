module Llama
  # Stateful text facade. The canonical transcript is rebuilt before each turn;
  # this conservative strategy keeps stop-truncated text and native KV state in
  # agreement until incremental token reconciliation is introduced.
  class Session
    include NativeResource

    def initialize(@model : Model, context_options : ContextOptions = ContextOptions.new)
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
      @transcript = candidate + result.text
      result
    ensure
      end_operation! if operation_started
    end

    def generate(prompt : String, options : GenerationOptions = GenerationOptions.new) : Generation
      generate(prompt, options) { |_chunk| }
    end

    def used_tokens : Int32
      ensure_open!
      Tokenizer.new(@model.vocab).encode(@transcript).size
    end

    def reset : Nil
      @mutex.synchronize do
        ensure_open!
        raise BusyError.new(self.class.to_s) if @running
        @context.memory.clear
        @transcript = ""
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
