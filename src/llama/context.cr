require "./context/error"

module Llama
  # Wrapper for the llama_context structure
  class Context
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
    #
    # Raises:
    # - Llama::Context::Error if the context cannot be created.
    def initialize(
      model : Model,
      n_ctx : UInt32 = 0,          # The maximum context size (0 = use model default)
      n_batch : UInt32 = 512,      # The maximum batch size
      n_threads : Int32 = 0,       # Number of threads for generation
      n_threads_batch : Int32 = 0, # Number of threads for batch processing
      embeddings : Bool = false,   # Enable or disable embeddings
      offload_kqv : Bool = false,  # Offload KQV to GPU
    )
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
      params.op_offload = false
      params.swa_full = true
      @handle = LibLlama.llama_init_from_model(model.to_unsafe, params)

      if @handle.null?
        Llama.unregister_context
        error_msg = Llama.format_error(
          "Failed to create context",
          -4, # Context creation error
          "n_ctx: #{n_ctx}, n_batch: #{n_batch}, n_threads: #{n_threads}, n_threads_batch: #{n_threads_batch}, embeddings: #{embeddings}, offload_kqv: #{offload_kqv}"
        )
        raise Context::Error.new(error_msg)
      end

      @model = model
      @adapters_lora = [] of AdapterLora
      @adapter_lora_scales = [] of Float32
    end

    private def sync_adapters_lora! : Int32
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

    # Returns the memory for this context (modern API)
    #
    # The memory system provides unified access to various memory types:
    # - Standard KV cache (llama_kv_cache_unified)
    # - SWA (Sliding Window Attention) cache
    # - Recurrent layer memory
    # - Hybrid attention/recurrent models
    #
    # Returns:
    # - A Memory instance
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
    # This can be called manually to release resources before garbage collection
    private def cleanup
      # Free the context handle
      if @handle && !@handle.null?
        LibLlama.llama_free(@handle)
        @handle = Pointer(LibLlama::LlamaContext).null
        Llama.unregister_context
      end
    end

    # Generates a response in a chat conversation
    #
    # Parameters:
    # - messages: Array of chat messages
    # - max_tokens: Maximum number of tokens to generate
    # - temperature: Sampling temperature
    # - template: Optional chat template (nil to use model's default)
    #
    # Returns:
    # - The generated response text
    #
    # Raises:
    # - ArgumentError if parameters are invalid
    # - Llama::Context::Error if text generation fails
    # - Llama::TokenizationError if the prompt cannot be tokenized
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

    # High-level batch processing methods

    # Process a sequence of tokens
    #
    # Parameters:
    # - tokens: Array of token IDs to process
    # - compute_logits_for_last: Whether to compute logits only for the last token
    # - seq_ids: Sequence IDs to use for all tokens
    # - n_seq_max: Maximum number of sequence IDs per token (default: 8)
    #
    # Returns:
    # - The result of the decode operation (0 on success)
    #
    # Raises:
    # - Llama::Batch::Error on error
    def process_tokens(tokens : Array(Int32), compute_logits_for_last : Bool = true, seq_ids : Array(Int32)? = nil, n_seq_max : Int32 = 8) : Int32
      if tokens.empty?
        raise ArgumentError.new("tokens array cannot be empty")
      end

      decode_tokens(tokens, compute_logits_for_last, seq_ids, n_seq_max)
    end

    # Process multiple prompts in batch
    #
    # Parameters:
    # - prompts: Array of text prompts to process
    # - compute_logits_for_last: Whether to compute logits only for the last token of each prompt
    #
    # Returns:
    # - Array of decode operation results (0 on success)
    #
    # Raises:
    # - Llama::Batch::Error on error
    # - Llama::TokenizationError if a prompt cannot be tokenized
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
              -6, # Tokenization error
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
            -3, # Batch processing error
            "prompt index: #{i}, error: #{ex.message}"
          )
          raise Batch::Error.new(error_msg)
        end
      end

      results
    end

    # Process embeddings
    #
    # Parameters:
    # - embeddings: Array of embedding vectors
    # - seq_ids: Sequence IDs to use for all embeddings
    # - n_seq_max: Maximum number of sequence IDs per token (default: 8)
    #
    # Returns:
    # - The result of the decode operation (0 on success)
    #
    # Raises:
    # - Llama::Batch::Error on error
    def process_embeddings(embeddings : Array(Array(Float32)), seq_ids : Array(Int32)? = nil, n_seq_max : Int32 = 8) : Int32
      if embeddings.empty?
        raise ArgumentError.new("embeddings array cannot be empty")
      end

      batch = Batch.for_embeddings(embeddings, seq_ids, n_seq_max)
      decode(batch)
    end

    # Creates a token batch with absolute token positions.
    #
    # Parameters:
    # - tokens: Tokens to process
    # - pos_offset: Absolute position of the first token
    # - compute_logits_for_last: Whether to compute logits only for the last token
    # - compute_logits: Whether to compute logits for this batch
    #
    # Returns:
    # - A prepared Batch ready for processing
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
    #
    # llama_decode rejects batches larger than the context's logical batch size.
    # Prompt prefill can still be longer than n_batch, so split tokens while keeping
    # absolute positions continuous and requesting logits only for the final token.
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
          -10, # Invalid parameter error
          "tokens: #{tokens.size}, n_ctx: #{context_size}"
        )
        raise Context::Error.new(error_msg)
      end

      batch_size = n_batch.to_i
      if batch_size <= 0
        error_msg = Llama.format_error(
          "Invalid batch size",
          -10, # Invalid parameter error
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
    #
    # Parameters:
    # - prompt: The input prompt
    # - sampler: The sampler chain to use
    # - max_tokens: Maximum number of tokens to generate (must be positive)
    #
    # Returns:
    # - The generated text
    #
    # Raises:
    # - ArgumentError if parameters are invalid
    # - Llama::Context::Error if text generation fails
    # - Llama::TokenizationError if the prompt cannot be tokenized
    # - Llama::Sampler::Error if sampling fails
    def generate_with_sampler(prompt : String, sampler : SamplerChain, max_tokens : Int32 = 128) : String
      # Validate parameters
      if max_tokens <= 0
        raise ArgumentError.new("max_tokens must be positive")
      end

      sampler.reset

      # Use the internal generation method with a custom token sampler
      generate_internal(prompt, max_tokens) do |_logits|
        begin
          # Sample the next token using the sampler chain
          token = sampler.sample(self)

          # Accept the token
          sampler.accept(token)

          token
        rescue ex
          error_msg = Llama.format_error(
            "Sampling failed",
            -9, # Sampling error
            ex.message
          )
          raise Sampler::Error.new(error_msg)
        end
      end
    end

    # Processes a batch of tokens with the encoder part of the model
    #
    # This function is used for encoder-decoder models to encode the input
    # before generating text with the decoder.
    #
    # Parameters:
    # - batch: The batch to process (can be a LibLlama::LlamaBatch or a Batch instance)
    #
    # Returns:
    # - 0 on success
    # - < 0 on error
    #
    # Raises:
    # - Llama::Batch::Error on error
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

    # Processes a batch of tokens with the decoder part of the model
    #
    # Parameters:
    # - batch: The batch to process (can be a LibLlama::LlamaBatch or a Batch instance)
    #
    # Returns:
    # - 0 on success
    # - 1 if no KV slot was found for the batch
    # - < 0 on error
    #
    # Raises:
    # - Llama::Batch::Error on error
    def decode(batch : LibLlama::LlamaBatch | Batch) : Int32
      batch_ptr = batch.is_a?(Batch) ? batch.to_unsafe : batch
      batch_size = n_batch.to_i

      validate_batch_for_decode!(batch_ptr, batch_size, n_ctx.to_i)

      result = LibLlama.llama_decode(@handle, batch_ptr)

      if result < 0
        error_msg = Llama.format_error(
          "Failed to decode batch",
          result,
          "batch size: #{batch_ptr.n_tokens}"
        )
        raise Batch::Error.new(error_msg)
      end

      result
    end

    private def validate_batch_for_decode!(batch : LibLlama::LlamaBatch, batch_size : Int32, context_size : Int32) : Nil
      if batch.n_tokens <= 0
        error_msg = Llama.format_error(
          "Cannot decode empty batch",
          -3,
          "n_tokens: #{batch.n_tokens}"
        )
        raise Batch::Error.new(error_msg)
      end

      if batch.n_tokens > batch_size
        error_msg = Llama.format_error(
          "Batch exceeds n_batch",
          -3, # Batch processing error
          "batch size: #{batch.n_tokens}, n_batch: #{batch_size}"
        )
        raise Batch::Error.new(error_msg)
      end

      unless batch.pos.null?
        batch.n_tokens.times do |i|
          pos = batch.pos[i]
          if pos < 0 || pos >= context_size
            error_msg = Llama.format_error(
              "Token position out of bounds",
              -10,
              "position #{pos} at index #{i}, n_ctx: #{context_size}"
            )
            raise Batch::Error.new(error_msg)
          end
        end
      end

      if !batch.n_seq_id.null? && !batch.seq_id.null?
        batch.n_tokens.times do |i|
          n_seq = batch.n_seq_id[i]
          if n_seq <= 0
            error_msg = Llama.format_error(
              "Invalid sequence ID count",
              -10,
              "n_seq_id[#{i}] = #{n_seq}"
            )
            raise Batch::Error.new(error_msg)
          end

          if batch.seq_id[i].null?
            error_msg = Llama.format_error(
              "Null sequence ID pointer",
              -10,
              "seq_id[#{i}] is null"
            )
            raise Batch::Error.new(error_msg)
          end

          n_seq.times do |j|
            seq_id = batch.seq_id[i][j]
            if seq_id < 0
              error_msg = Llama.format_error(
                "Invalid sequence ID",
                -10,
                "seq_id[#{i}][#{j}] = #{seq_id}"
              )
              raise Batch::Error.new(error_msg)
            end
          end
        end
      end

      unless batch.logits.null?
        batch.n_tokens.times do |i|
          logits = batch.logits[i]
          next if logits == 0 || logits == 1

          error_msg = Llama.format_error(
            "Invalid logits flag",
            -10,
            "logits[#{i}] = #{logits}"
          )
          raise Batch::Error.new(error_msg)
        end
      end
    end

    # Gets the logits for a specific output index.
    #
    # Parameters:
    # - i: Output index (negative indices access from the end; -1 is latest)
    #
    # Returns:
    # - A pointer to the logits array for the specified output, or nil if unavailable
    def logits_ith(i : Int32) : Pointer(Float32)?
      ptr = LibLlama.llama_get_logits_ith(@handle, i)
      ptr.null? ? nil : ptr
    end

    # Gets the logits for the latest output token.
    #
    # Returns:
    # - A pointer to the logits array
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
    #
    # Parameters:
    # - prompt: The input prompt
    # - max_tokens: Maximum number of tokens to generate (must be positive)
    # - temperature: Sampling temperature (0.0 = greedy, 1.0 = more random)
    #
    # Returns:
    # - The generated text
    #
    # Raises:
    # - ArgumentError if parameters are invalid
    # - Llama::Context::Error if text generation fails
    # - Llama::TokenizationError if the prompt cannot be tokenized
    def generate(prompt : String, max_tokens : Int32 = 128, temperature : Float32 = 0.8) : String
      # Validate parameters
      if max_tokens <= 0
        raise ArgumentError.new("max_tokens must be positive")
      end

      if temperature < 0
        raise ArgumentError.new("temperature must be non-negative")
      end

      # 空プロンプトも許可する

      # Use the internal generation method with temperature sampling
      generate_internal(prompt, max_tokens) do |logits|
        sample_token(logits, temperature)
      end
    end

    # Internal implementation of text generation
    #
    # Parameters:
    # - prompt: The input prompt
    # - max_tokens: Maximum number of tokens to generate
    # - &token_sampler: Block that samples the next token given logits
    #
    # Returns:
    # - The generated text
    #
    # Yields:
    # - logits: Pointer to logits array
    #
    # Raises:
    # - ArgumentError if input tokens are empty
    # - Llama::TokenizationError if the prompt cannot be tokenized
    # - Llama::Batch::Error if batch processing fails
    # - Llama::Context::Error if text generation fails
    private def generate_internal(prompt : String, max_tokens : Int32, &token_sampler : Pointer(Float32) -> Int32) : String
      # Tokenize the prompt
      begin
        input_tokens = @model.vocab.tokenize(prompt)
      rescue ex
        error_msg = Llama.format_error(
          "Failed to tokenize prompt",
          -6, # Tokenization error
          ex.message
        )
        raise Llama::TokenizationError.new(error_msg)
      end

      # Ensure input tokens are not empty
      if input_tokens.empty?
        error_msg = Llama.format_error(
          "Tokenization resulted in empty token array",
          -6, # Tokenization error
          "prompt length: #{prompt.size}"
        )
        raise Llama::TokenizationError.new(error_msg)
      end

      # Initialize the result string
      output_tokens = [] of Int32

      # Current position in the sequence
      pos = input_tokens.size

      # High-level generation starts a fresh sequence. Stateful continuation is
      # available through the lower-level decode/process_tokens APIs.
      memory.clear

      # Process the prompt first. Long prompts are split by n_batch, while
      # prompts beyond n_ctx are rejected before llama_decode can fail.
      decode_prompt(input_tokens)

      context_size = n_ctx.to_i

      # Generate up to max_tokens
      max_tokens.times do |i|
        begin
          break if pos > context_size

          # Get the logits for the last token
          logits = self.logits

          # Sample the next token using the provided sampler
          next_token = token_sampler.call(logits)

          # Check for end-of-generation tokens such as EOS or EOT.
          break if @model.vocab.eog?(next_token)

          output_tokens << next_token

          token_pos = pos
          pos += 1
          break if i == max_tokens - 1 || token_pos >= context_size

          # Process the generated token so the next iteration can sample from it
          decode(generated_token_batch(next_token, token_pos))
        rescue ex : Batch::Error | Llama::TokenizationError
          raise ex
        rescue ex
          error_msg = Llama.format_error(
            "Text generation failed",
            nil,
            "at token position: #{pos}, error: #{ex.message}"
          )
          raise Context::Error.new(error_msg)
        end
      end

      @model.vocab.detokenize(output_tokens)
    end

    # Samples a token based on logits and temperature
    #
    # Parameters:
    # - logits: Pointer to logits array
    # - temperature: Sampling temperature (0.0 = greedy, >0.0 = random sampling)
    #
    # Returns:
    # - The sampled token ID
    #
    # Raises:
    # - Llama::Sampler::Error if sampling fails
    private def sample_token(logits : Pointer(Float32), temperature : Float32) : Int32
      sample_token_impl(logits, temperature)
    rescue ex : Sampler::Error
      raise ex
    rescue ex
      error_msg = Llama.format_error(
        "Token sampling failed",
        -9, # Sampling error
        "temperature: #{temperature}, error: #{ex.message}"
      )
      raise Sampler::Error.new(error_msg)
    end

    private def sample_token_impl(logits : Pointer(Float32), temperature : Float32) : Int32
      if temperature <= 0.0
        # Greedy sampling - just pick the most likely token
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

      # Apply temperature without mutating the context-owned logits buffer.
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
          -9, # Sampling error
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
      token = n_vocab - 1 # Default to last token

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
    #
    # This method prints performance statistics about the context to STDERR.
    # It's useful for debugging and performance analysis.
    def print_perf
      LibLlama.llama_perf_context_print(@handle)
    end

    # Reset performance counters for this context
    #
    # This method resets all performance counters for the context.
    def reset_perf
      LibLlama.llama_perf_context_reset(@handle)
    end

    # NOTE: llama_memory_breakdown_print was removed in llama.cpp b9297.
    def print_memory_breakdown
      raise Error.new("print_memory_breakdown is not supported on llama.cpp b9297+")
    end

    # Attaches a LoRA adapter to this context
    #
    # Parameters:
    # - adapter: The LoRA adapter to attach
    # - scale: Scaling factor for the adapter (default: 1.0)
    #
    # Returns:
    # - 0 on success, non-zero on error
    #
    # Raises:
    # - Llama::Context::Error if the adapter cannot be attached
    def attach_adapter_lora(adapter : AdapterLora, scale : Float32 = 1.0) : Int32
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

    # Detaches a LoRA adapter from this context
    #
    # Parameters:
    # - adapter: The LoRA adapter to detach
    #
    # Returns:
    # - 0 on success, non-zero on error
    #
    # Raises:
    # - Llama::Context::Error if the adapter cannot be detached
    def detach_adapter_lora(adapter : AdapterLora) : Int32
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

    # Clears all LoRA adapters from this context
    def clear_adapters_lora
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

    # Applies a control vector to the LoRA adapter
    #
    # Parameters:
    # - data: The control vector data
    # - n_embd: Embedding dimension per layer
    # - il_start: Start layer index (inclusive, 1-based)
    # - il_end: End layer index (inclusive, 1-based)
    #
    # Returns:
    # - 0 on success, non-zero on error
    #
    # Raises:
    # - Llama::Context::Error if the control vector cannot be applied
    def apply_adapter_cvec(data : Slice(Float32), n_embd : Int32, il_start : Int32, il_end : Int32) : Int32
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

    # Sets whether the model is in embeddings mode or not
    # If true, embeddings will be returned but logits will not
    #
    # Parameters:
    # - enabled: Whether to enable embeddings mode
    def embeddings=(enabled : Bool)
      LibLlama.llama_set_embeddings(@handle, enabled)
    end

    # Gets the pooling type used for embeddings
    #
    # Returns:
    # - The pooling type as a PoolingType enum
    def pooling_type : LibLlama::LlamaPoolingType
      LibLlama.llama_pooling_type(@handle)
    end

    # Gets embeddings for the latest output token.
    #
    # Internally this uses the index-based embeddings API.
    #
    # Returns:
    # - An array of embeddings, or nil if embeddings are not available
    #
    # Raises:
    # - Llama::Context::Error if embeddings mode is not enabled
    def embeddings : Array(Float32)?
      get_embeddings_ith(-1)
    end

    # Gets the embeddings for a specific token
    #
    # Parameters:
    # - i: The token index (negative indices can be used to access in reverse order)
    #
    # Returns:
    # - An array of embedding values, or nil if not available
    #
    # Raises:
    # - Llama::Context::Error if embeddings mode is not enabled
    def get_embeddings_ith(i : Int32) : Array(Float32)?
      ptr = LibLlama.llama_get_embeddings_ith(@handle, i)
      return if ptr.null?

      # Get the embedding dimension from the model
      n_embd = @model.n_embd

      # Copy the embeddings to a Crystal array
      result = Array(Float32).new(n_embd)
      n_embd.times do |j|
        result << ptr[j]
      end

      result
    end

    # Gets the embeddings for a specific sequence
    #
    # Parameters:
    # - seq_id: The sequence ID
    #
    # Returns:
    # - An array of embedding values, or nil if not available
    #
    # Raises:
    # - Llama::Context::Error if embeddings mode is not enabled
    def get_embeddings_seq(seq_id : Int32) : Array(Float32)?
      ptr = LibLlama.llama_get_embeddings_seq(@handle, seq_id)
      return if ptr.null?

      # Get the embedding dimension from the model
      n_embd = @model.n_embd

      # Copy the embeddings to a Crystal array
      result = Array(Float32).new(n_embd)
      n_embd.times do |i|
        result << ptr[i]
      end

      result
    end

    # Applies the chat template to the given messages and returns the formatted prompt.
    #
    # Parameters:
    # - messages: Array of ChatMessage (user/assistant/system)
    # - add_assistant: Whether to add assistant role (default: true)
    # - template: Optional template string (default: model's template)
    #
    # Returns:
    # - The formatted prompt string.
    def apply_chat_template(messages : Array(ChatMessage), add_assistant : Bool = true, template : String? = nil) : String
      tmpl = template || @model.chat_template || ""
      Llama.apply_chat_template(tmpl, messages, add_assistant)
    end

    @handle : LibLlama::LlamaContext*
    @model : Model

    # :nodoc:
    def clone
      raise NotImplementedError.new("clone is not supported for #{self.class}")
    end

    # :nodoc:
    def dup
      raise NotImplementedError.new("dup is not supported for #{self.class}")
    end
  end
end
