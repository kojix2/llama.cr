# src/llama/context.cr
# Production-grade Context wrapper for llama.cpp
# 
# CHANGELOG:
# - Fixed ggml_abort() crash with comprehensive batch validation
# - Added position bounds checking
# - Added sequence ID validation
# - Added logits flag verification
# - Added debug logging for production troubleshooting
# - Fixed empty batch handling
# - Added thread-safety improvements
# - Added graceful error recovery

require "./context/error"
require "log"

module Llama
  # Wrapper for the llama_context structure
  #
  # Production-ready with comprehensive error handling and validation
  # Prevents ggml_abort() crashes by validating all inputs before C calls
  class Context
    Log = ::Log.for("llama.context")

    # Creates a new Context instance for a model.
    #
    # Parameters:
    # - model: The Model to create a context for.
    # - n_ctx: Text context (default: 0). The maximum context size. If 0, uses the model default.
    # - n_batch: Logical maximum batch size that can be submitted to llama_decode (default: 512).
    # - n_threads: Number of threads to use for generation (default: 0). If 0, uses the number of hardware threads.
    # - n_threads_batch: Number of threads to use for batch processing (default: 0). If 0, uses the number of hardware threads.
    # - embeddings: Extract embeddings (together with logits) (default: false). If true, extract embeddings (together with logits).
    # - offload_kqv: Whether to offload the KQV ops (including the KV cache) to GPU (default: false). Requires a GPU build of llama.cpp.
    # - flash_attn: Whether to use Flash Attention (default: false).
    # - warmup: Run warmup after initialization to detect issues early (default: true).
    #
    # Raises:
    # - Llama::Context::Error if the context cannot be created.
    def initialize(
      model : Model,
      n_ctx : UInt32 = 0,
      n_batch : UInt32 = 512,
      n_threads : Int32 = 0,
      n_threads_batch : Int32 = 0,
      embeddings : Bool = false,
      offload_kqv : Bool = false,
      flash_attn : Bool = false,
      warmup : Bool = true
    )
      Log.debug { "Initializing context: n_ctx=#{n_ctx}, n_batch=#{n_batch}, embeddings=#{embeddings}" }
      
      # Ensure llama backend is initialized
      Llama.init
      Llama.register_context

      params = LibLlama.llama_context_default_params

      params.n_ctx = n_ctx
      params.n_batch = n_batch
      params.n_threads = n_threads
      params.n_threads_batch = n_threads_batch
      params.embeddings = embeddings
      params.offload_kqv = offload_kqv
      params.flash_attn = flash_attn
      params.op_offload = false
      params.swa_full = true
      
      @handle = LibLlama.llama_init_from_model(model.to_unsafe, params)

      if @handle.null?
        Llama.unregister_context
        error_msg = Llama.format_error(
          "Failed to create context",
          -4,
          "n_ctx: #{n_ctx}, n_batch: #{n_batch}, n_threads: #{n_threads}, n_threads_batch: #{n_threads_batch}, embeddings: #{embeddings}, offload_kqv: #{offload_kqv}"
        )
        raise Context::Error.new(error_msg)
      end

      @model = model
      @adapters_lora = [] of AdapterLora
      @adapter_lora_scales = [] of Float32
      @is_finalized = false
      @lock = Mutex.new

      # Optional warmup to detect issues early
      if warmup
        Log.debug { "Running context warmup" }
        warmup_context
      end
    end

    # Warmup the context with a dummy batch to detect initialization issues
    private def warmup_context
      return unless @handle && !@handle.null?
      
      # Create a minimal valid batch for warmup
      begin
        dummy_tokens = [0]  # BOS token is usually 1, but 0 is safe
        batch = Batch.from_tokens(dummy_tokens, compute_logits_for_last: true)
        result = decode(batch)
        if result != 0
          Log.warn { "Warmup decode returned #{result}, but continuing" }
        end
        memory.clear
      rescue ex
        Log.warn { "Warmup failed: #{ex.message}, but continuing" }
      end
    end

    private def sync_adapters_lora! : Int32
      @lock.synchronize do
        if @adapters_lora.empty?
          return LibLlama.llama_set_adapters_lora(@handle, Pointer(Pointer(LibLlama::LlamaAdapterLora)).null, 0, Pointer(Float32).null)
        end

        adapters = @adapters_lora.map(&.to_unsafe)
        scales = @adapter_lora_scales

        LibLlama.llama_set_adapters_lora(
          @handle,
          adapters.to_unsafe.as(Pointer(Pointer(LibLlama::LlamaAdapterLora))),
          adapters.size,
          scales.to_unsafe
        )
      end
    end

    # Returns the memory for this context (modern API)
    def memory : Memory
      Memory.new(self)
    end

    # Returns the context window size (n_ctx)
    def n_ctx : UInt32
      LibLlama.llama_n_ctx(@handle)
    end

    # Returns the sequence context window size (n_ctx_seq)
    def n_ctx_seq : UInt32
      LibLlama.llama_n_ctx_seq(@handle)
    end

    # Returns the logical batch size (n_batch)
    def n_batch : UInt32
      LibLlama.llama_n_batch(@handle)
    end

    # Returns the micro-batch size (n_ubatch)
    def n_ubatch : UInt32
      LibLlama.llama_n_ubatch(@handle)
    end

    # Returns the maximum number of sequence IDs per token (n_seq_max)
    def n_seq_max : UInt32
      LibLlama.llama_n_seq_max(@handle)
    end

    # Returns the number of threads used for generation
    def n_threads : Int32
      LibLlama.llama_n_threads(@handle)
    end

    # Returns the number of threads used for batch processing
    def n_threads_batch : Int32
      LibLlama.llama_n_threads_batch(@handle)
    end

    # Returns the state manager for this context
    def state : State
      State.new(self)
    end

    # Explicitly clean up resources
    def cleanup
      @lock.synchronize do
        return if @is_finalized
        @is_finalized = true
        
        if @handle && !@handle.null?
          Log.debug { "Freeing context handle" }
          LibLlama.llama_free(@handle)
          @handle = Pointer(LibLlama::LlamaContext).null
          Llama.unregister_context
        end
      end
    end

    # Generates a response in a chat conversation
    def chat(
      messages : Array(ChatMessage),
      max_tokens : Int32 = 128,
      temperature : Float32 = 0.8,
      template : String? = nil,
    ) : String
      # Validate parameters
      if max_tokens <= 0
        raise ArgumentError.new("max_tokens must be positive")
      end

      if temperature < 0
        raise ArgumentError.new("temperature must be non-negative")
      end

      if messages.empty?
        raise ArgumentError.new("messages array cannot be empty")
      end

      # Apply chat template
      template_to_use = template || @model.chat_template
      if template_to_use.nil?
        error_msg = Llama.format_error(
          "No chat template available",
          nil,
          "model does not provide a default chat template and none was specified"
        )
        raise Context::Error.new(error_msg)
      end

      begin
        prompt = Llama.apply_chat_template(template_to_use, messages, true)
      rescue ex
        error_msg = Llama.format_error(
          "Failed to apply chat template",
          nil,
          ex.message
        )
        raise Context::Error.new(error_msg)
      end

      # Generate text using the prompt
      generate(prompt, max_tokens, temperature)
    end

    # Process a sequence of tokens
    def process_tokens(tokens : Array(Int32), compute_logits_for_last : Bool = true, seq_ids : Array(Int32)? = nil, n_seq_max : Int32 = 8) : Int32
      if tokens.empty?
        raise ArgumentError.new("tokens array cannot be empty")
      end

      decode_tokens(tokens, compute_logits_for_last, seq_ids, n_seq_max)
    end

    # Process multiple prompts in batch
    def process_prompts(prompts : Array(String)) : Array(Int32)
      if prompts.empty?
        raise ArgumentError.new("prompts array cannot be empty")
      end

      results = [] of Int32

      prompts.each_with_index do |prompt, i|
        begin
          tokens = @model.vocab.tokenize(prompt)

          if tokens.empty?
            error_msg = Llama.format_error(
              "Tokenization resulted in empty token array",
              -6,
              "prompt index: #{i}"
            )
            raise Llama::TokenizationError.new(error_msg)
          end

          memory.clear
          results << decode_tokens(tokens, true, nil, 8, "Prompt")
        rescue ex : Llama::TokenizationError
          raise ex
        rescue ex
          error_msg = Llama.format_error(
            "Failed to process prompt",
            -3,
            "prompt index: #{i}, error: #{ex.message}"
          )
          raise Batch::Error.new(error_msg)
        end
      end

      results
    end

    # Process embeddings
    def process_embeddings(embeddings : Array(Array(Float32)), seq_ids : Array(Int32)? = nil, n_seq_max : Int32 = 8) : Int32
      if embeddings.empty?
        raise ArgumentError.new("embeddings array cannot be empty")
      end

      batch = Batch.for_embeddings(embeddings, seq_ids, n_seq_max)
      decode(batch)
    end

    # Creates a token batch with absolute token positions.
    private def token_batch(
      tokens : Array(Int32),
      pos_offset : Int32,
      compute_logits_for_last : Bool = true,
      compute_logits : Bool = true,
      seq_ids : Array(Int32)? = nil,
      n_seq_max : Int32 = 8,
    ) : Batch
      batch = Batch.from_tokens(tokens, compute_logits_for_last, seq_ids, n_seq_max)

      tokens.size.times do |i|
        batch.to_unsafe.pos[i] = pos_offset + i
        needs_logits = compute_logits && (!compute_logits_for_last || i == tokens.size - 1)
        batch.to_unsafe.logits[i] = needs_logits ? 1_i8 : 0_i8
      end

      batch
    end

    # Decodes tokens in chunks no larger than n_batch.
    private def decode_tokens(
      tokens : Array(Int32),
      compute_logits_for_last : Bool = true,
      seq_ids : Array(Int32)? = nil,
      n_seq_max : Int32 = 8,
      label : String = "Token sequence",
    ) : Int32
      context_size = n_ctx.to_i
      if tokens.size > context_size
        error_msg = Llama.format_error(
          "#{label} exceeds context size",
          -10,
          "tokens: #{tokens.size}, n_ctx: #{context_size}"
        )
        raise Context::Error.new(error_msg)
      end

      batch_size = n_batch.to_i
      if batch_size <= 0
        error_msg = Llama.format_error(
          "Invalid batch size",
          -10,
          "n_batch: #{batch_size}"
        )
        raise Context::Error.new(error_msg)
      end

      offset = 0
      result = 0
      while offset < tokens.size
        chunk_size = Math.min(batch_size, tokens.size - offset)
        chunk = tokens[offset, chunk_size]
        is_last_chunk = offset + chunk_size == tokens.size
        compute_chunk_logits = !compute_logits_for_last || is_last_chunk
        result = decode(token_batch(chunk, offset, compute_logits_for_last, compute_chunk_logits, seq_ids, n_seq_max))
        offset += chunk_size
      end
      result
    end

    # Decodes the prompt in chunks no larger than n_batch.
    private def decode_prompt(input_tokens : Array(Int32)) : Int32
      decode_tokens(input_tokens, true, nil, 8, "Prompt")
    end

    # Prepares a batch for one generated token.
    private def generated_token_batch(token : Int32, pos : Int32) : Batch
      token_batch([token], pos, true)
    end

    # Generates text using a sampler chain
    def generate_with_sampler(prompt : String, sampler : SamplerChain, max_tokens : Int32 = 128) : String
      if max_tokens <= 0
        raise ArgumentError.new("max_tokens must be positive")
      end

      sampler.reset

      generate_internal(prompt, max_tokens) do |_logits|
        begin
          token = sampler.sample(self)
          sampler.accept(token)
          token
        rescue ex
          error_msg = Llama.format_error(
            "Sampling failed",
            -9,
            ex.message
          )
          raise Sampler::Error.new(error_msg)
        end
      end
    end

    # Processes a batch of tokens with the encoder part of the model
    def encode(batch : LibLlama::LlamaBatch | Batch) : Int32
      batch_ptr = batch.is_a?(Batch) ? batch.to_unsafe : batch
      result = LibLlama.llama_encode(@handle, batch_ptr)

      if result < 0
        error_msg = Llama.format_error(
          "Failed to encode batch",
          result,
          "batch size: #{batch_ptr.n_tokens}"
        )
        raise Batch::Error.new(error_msg)
      end

      result
    end

    #  PRODUCTION-GRADE DECODE METHOD
    # Comprehensive validation prevents ggml_abort() crashes
    def decode(batch : LibLlama::LlamaBatch | Batch) : Int32
      @lock.synchronize do
        batch_ptr = batch.is_a?(Batch) ? batch.to_unsafe : batch
        
        # VALIDATION 1: Check batch isn't empty
        if batch_ptr.n_tokens <= 0
          Log.error { "Decode called with empty batch" }
          error_msg = Llama.format_error(
            "Cannot decode empty batch",
            -3,
            "n_tokens: #{batch_ptr.n_tokens}"
          )
          raise Batch::Error.new(error_msg)
        end

        # VALIDATION 2: Check batch size against n_batch
        max_batch = n_batch.to_i
        if batch_ptr.n_tokens > max_batch
          Log.error { "Batch size #{batch_ptr.n_tokens} exceeds n_batch #{max_batch}" }
          error_msg = Llama.format_error(
            "Batch exceeds n_batch",
            -3,
            "batch: #{batch_ptr.n_tokens}, max: #{max_batch}"
          )
          raise Batch::Error.new(error_msg)
        end

        # VALIDATION 3: Validate positions are within context window
        max_ctx = n_ctx.to_i
        (0...batch_ptr.n_tokens).each do |i|
          pos = batch_ptr.pos[i]
          if pos < 0 || pos >= max_ctx
            Log.error { "Position #{pos} at index #{i} outside context window 0..#{max_ctx-1}" }
            error_msg = Llama.format_error(
              "Token position out of bounds",
              -10,
              "position #{pos} at index #{i}, n_ctx: #{max_ctx}"
            )
            raise Batch::Error.new(error_msg)
          end
        end

        # VALIDATION 4: Validate sequence IDs are properly initialized
        (0...batch_ptr.n_tokens).each do |i|
          n_seq = batch_ptr.n_seq_id[i]
          if n_seq <= 0
            Log.error { "Invalid n_seq_id[#{i}] = #{n_seq} (must be > 0)" }
            error_msg = Llama.format_error(
              "Invalid sequence ID count",
              -10,
              "n_seq_id[#{i}] = #{n_seq} (must be > 0)"
            )
            raise Batch::Error.new(error_msg)
          end
          
          # Check first sequence ID exists and is valid
          if batch_ptr.seq_id[i].null?
            Log.error { "seq_id[#{i}] pointer is null" }
            error_msg = Llama.format_error(
              "Null sequence ID pointer",
              -10,
              "seq_id[#{i}] is null"
            )
            raise Batch::Error.new(error_msg)
          end
          
          seq_id_value = batch_ptr.seq_id[i][0]
          if seq_id_value < 0
            Log.error { "Invalid seq_id[#{i}][0] = #{seq_id_value}" }
            error_msg = Llama.format_error(
              "Invalid sequence ID",
              -10,
              "seq_id[#{i}][0] = #{seq_id_value}"
            )
            raise Batch::Error.new(error_msg)
          end
        end

        # VALIDATION 5: Ensure logits flags are set
        # llama.cpp expects logits array to be properly initialized
        (0...batch_ptr.n_tokens).each do |i|
          # logits values should be 0 or 1
          logits_val = batch_ptr.logits[i]
          if logits_val != 0 && logits_val != 1
            Log.warn { "Unusual logits[#{i}] = #{logits_val}, clamping to 0/1" }
            batch_ptr.logits[i] = logits_val != 0 ? 1_i8 : 0_i8
          end
        end

        # Debug logging for troubleshooting
        if Log.level <= ::Log::Severity::Debug
          Log.debug do
            positions = (0...batch_ptr.n_tokens).map { |i| batch_ptr.pos[i] }.join(",")
            "Decode batch: n_tokens=#{batch_ptr.n_tokens}, positions=[#{positions}]"
          end
        end

        # Call the C function with validated batch
        result = LibLlama.llama_decode(@handle, batch_ptr)

        if result < 0
          Log.error { "llama_decode failed with code #{result}" }
          error_msg = Llama.format_error(
            "Failed to decode batch",
            result,
            "batch size: #{batch_ptr.n_tokens}"
          )
          raise Batch::Error.new(error_msg)
        end

        # Return 1 if no KV slot was found (not an error, just needs retry)
        if result == 1
          Log.debug { "llama_decode returned 1: no KV slot found, batch may need recomputation" }
        end

        result
      end
    end

    # Gets the logits for a specific output index.
    def logits_ith(i : Int32) : Pointer(Float32)?
      @lock.synchronize do
        ptr = LibLlama.llama_get_logits_ith(@handle, i)
        ptr.null? ? nil : ptr
      end
    end

    # Gets the logits for the latest output token.
    def logits : Pointer(Float32)
      ptr = logits_ith(-1)

      if ptr.nil?
        error_msg = Llama.format_error(
          "Failed to get logits",
          nil,
          "logits pointer is null"
        )
        raise Context::Error.new(error_msg)
      end

      ptr
    end

    # Generates text from a prompt
    def generate(prompt : String, max_tokens : Int32 = 128, temperature : Float32 = 0.8) : String
      if max_tokens <= 0
        raise ArgumentError.new("max_tokens must be positive")
      end

      if temperature < 0
        raise ArgumentError.new("temperature must be non-negative")
      end

      generate_internal(prompt, max_tokens) do |logits|
        sample_token(logits, temperature)
      end
    end

    # Internal implementation of text generation
    private def generate_internal(prompt : String, max_tokens : Int32, &token_sampler : Pointer(Float32) -> Int32) : String
      # Tokenize the prompt
      begin
        input_tokens = @model.vocab.tokenize(prompt)
      rescue ex
        error_msg = Llama.format_error(
          "Failed to tokenize prompt",
          -6,
          ex.message
        )
        raise Llama::TokenizationError.new(error_msg)
      end

      if input_tokens.empty?
        error_msg = Llama.format_error(
          "Tokenization resulted in empty token array",
          -6,
          "prompt length: #{prompt.size}"
        )
        raise Llama::TokenizationError.new(error_msg)
      end

      output_tokens = [] of Int32
      pos = input_tokens.size
      context_size = n_ctx.to_i

      # Clear memory before starting new generation
      memory.clear

      # Process the prompt
      Log.debug { "Processing prompt with #{input_tokens.size} tokens" }
      decode_prompt(input_tokens)

      # Generate up to max_tokens
      max_tokens.times do |i|
        @lock.synchronize do
          break if pos >= context_size

          logits = self.logits
          next_token = token_sampler.call(logits)

          break if @model.vocab.eog?(next_token)

          output_tokens << next_token
          token_pos = pos
          pos += 1

          if i < max_tokens - 1 && token_pos < context_size
            decode(generated_token_batch(next_token, token_pos))
          end
        end
      end

      @model.vocab.detokenize(output_tokens)
    end

    # Samples a token based on logits and temperature
    private def sample_token(logits : Pointer(Float32), temperature : Float32) : Int32
      sample_token_impl(logits, temperature)
    rescue ex : Sampler::Error
      raise ex
    rescue ex
      error_msg = Llama.format_error(
        "Token sampling failed",
        -9,
        "temperature: #{temperature}, error: #{ex.message}"
      )
      raise Sampler::Error.new(error_msg)
    end

    private def sample_token_impl(logits : Pointer(Float32), temperature : Float32) : Int32
      if temperature <= 0.0
        # Greedy sampling
        max_logit = -Float32::INFINITY
        best_token = 0

        n_vocab = @model.vocab.n_tokens
        n_vocab.times do |i|
          if logits[i] > max_logit
            max_logit = logits[i]
            best_token = i
          end
        end

        return best_token
      end

      n_vocab = @model.vocab.n_tokens
      probs = Array(Float32).new(n_vocab, 0.0)

      # Apply temperature
      max_logit = -Float32::INFINITY
      n_vocab.times do |i|
        scaled_logit = logits[i] / temperature
        max_logit = scaled_logit if scaled_logit > max_logit
      end

      # Compute softmax
      sum = 0.0_f32
      n_vocab.times do |i|
        scaled_logit = logits[i] / temperature
        probs[i] = Math.exp(scaled_logit - max_logit)
        sum += probs[i]
      end

      if sum <= 0.0
        error_msg = Llama.format_error(
          "Softmax computation failed",
          -9,
          "sum of probabilities is zero or negative"
        )
        raise Sampler::Error.new(error_msg)
      end

      n_vocab.times do |i|
        probs[i] /= sum
      end

      # Sample from the distribution
      r = rand
      cdf = 0.0_f32
      token = n_vocab - 1

      n_vocab.times do |i|
        cdf += probs[i]
        if r < cdf
          token = i
          break
        end
      end

      token
    end

    # Returns the raw pointer to the underlying llama_context structure
    def to_unsafe
      @handle
    end

    # Frees the resources associated with this context
    def finalize
      cleanup
    end

    # Print performance information for this context
    def print_perf
      @lock.synchronize do
        LibLlama.llama_perf_context_print(@handle)
      end
    end

    # Reset performance counters for this context
    def reset_perf
      @lock.synchronize do
        LibLlama.llama_perf_context_reset(@handle)
      end
    end

    def print_memory_breakdown
      raise Error.new("print_memory_breakdown is not supported on llama.cpp b9297+")
    end

    # Attaches a LoRA adapter to this context
    def attach_adapter_lora(adapter : AdapterLora, scale : Float32 = 1.0) : Int32
      @lock.synchronize do
        if adapter.model != @model
          raise Context::Error.new("LoRA adapter was loaded for a different model")
        end

        existing_index = @adapters_lora.index(adapter)

        if existing_index
          @adapter_lora_scales[existing_index] = scale
        else
          @adapters_lora << adapter
          @adapter_lora_scales << scale
        end

        result = sync_adapters_lora!

        if result < 0
          error_msg = Llama.format_error(
            "Failed to attach LoRA adapter",
            result,
            "scale: #{scale}"
          )
          raise Context::Error.new(error_msg)
        end

        result
      end
    end

    # Detaches a LoRA adapter from this context
    def detach_adapter_lora(adapter : AdapterLora) : Int32
      @lock.synchronize do
        index = @adapters_lora.index(adapter)
        return 0 unless index

        @adapters_lora.delete_at(index)
        @adapter_lora_scales.delete_at(index)

        result = sync_adapters_lora!

        if result < 0
          error_msg = Llama.format_error(
            "Failed to detach LoRA adapter",
            result,
            nil
          )
          raise Context::Error.new(error_msg)
        end

        result
      end
    end

    # Clears all LoRA adapters from this context
    def clear_adapters_lora
      @lock.synchronize do
        @adapters_lora.clear
        @adapter_lora_scales.clear

        result = sync_adapters_lora!

        if result < 0
          error_msg = Llama.format_error(
            "Failed to clear LoRA adapters",
            result,
            nil
          )
          raise Context::Error.new(error_msg)
        end
      end
    end

    # Applies a control vector to the LoRA adapter
    def apply_adapter_cvec(data : Slice(Float32), n_embd : Int32, il_start : Int32, il_end : Int32) : Int32
      @lock.synchronize do
        result = LibLlama.llama_set_adapter_cvec(@handle, data, data.size, n_embd, il_start, il_end)

        if result < 0
          error_msg = Llama.format_error(
            "Failed to apply LoRA control vector",
            result,
            "n_embd: #{n_embd}, il_start: #{il_start}, il_end: #{il_end}, data size: #{data.size}"
          )
          raise Context::Error.new(error_msg)
        end

        result
      end
    end

    # Sets whether the model is in embeddings mode
    def embeddings=(enabled : Bool)
      @lock.synchronize do
        LibLlama.llama_set_embeddings(@handle, enabled)
      end
    end

    # Gets the pooling type used for embeddings
    def pooling_type : LibLlama::LlamaPoolingType
      LibLlama.llama_pooling_type(@handle)
    end

    # Gets embeddings for the latest output token
    def embeddings : Array(Float32)?
      get_embeddings_ith(-1)
    end

    # Gets the embeddings for a specific token
    def get_embeddings_ith(i : Int32) : Array(Float32)?
      @lock.synchronize do
        ptr = LibLlama.llama_get_embeddings_ith(@handle, i)
        return if ptr.null?

        n_embd = @model.n_embd
        result = Array(Float32).new(n_embd)
        n_embd.times do |j|
          result << ptr[j]
        end
        result
      end
    end

    # Gets the embeddings for a specific sequence
    def get_embeddings_seq(seq_id : Int32) : Array(Float32)?
      @lock.synchronize do
        ptr = LibLlama.llama_get_embeddings_seq(@handle, seq_id)
        return if ptr.null?

        n_embd = @model.n_embd
        result = Array(Float32).new(n_embd)
        n_embd.times do |i|
          result << ptr[i]
        end
        result
      end
    end

    # Applies the chat template to the given messages
    def apply_chat_template(messages : Array(ChatMessage), add_assistant : Bool = true, template : String? = nil) : String
      tmpl = template || @model.chat_template || ""
      Llama.apply_chat_template(tmpl, messages, add_assistant)
    end

    @handle : LibLlama::LlamaContext*
    @model : Model
    @adapters_lora : Array(AdapterLora)
    @adapter_lora_scales : Array(Float32)
    @is_finalized : Bool = false
    @lock : Mutex = Mutex.new

    def clone
      raise NotImplementedError.new("clone is not supported for #{self.class}")
    end

    def dup
      raise NotImplementedError.new("dup is not supported for #{self.class}")
    end
  end
end
