require "./context/error"
require "./memory"

module Llama
  # Wrapper for the llama_context structure
  class Context
    # Creates a new Context instance for a model.
    #
    # Parameters:
    # - model: The Model to create a context for.
    # - n_ctx: Text context (default: 0). The maximum context size. If 0, a minimum context size of 512 is used.
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
      n_ctx : UInt32 = 0,          # The maximum context size (0 = use minimum of 512)
      n_batch : UInt32 = 512,      # The maximum batch size
      n_threads : Int32 = 0,       # Number of threads for generation
      n_threads_batch : Int32 = 0, # Number of threads for batch processing
      embeddings : Bool = false,   # Enable or disable embeddings
      offload_kqv : Bool = false,  # Offload KQV to GPU
    )
      # Ensure llama backend is initialized
      Llama.init

      params = LibLlama.llama_context_default_params

      # Ensure a minimum context size of 512 when n_ctx is 0
      # This helps with vocab_only models where the default context size might not be available
      actual_n_ctx = n_ctx == 0 ? 512_u32 : n_ctx

      params.n_ctx = actual_n_ctx
      params.n_batch = n_batch
      params.n_threads = n_threads
      params.n_threads_batch = n_threads_batch
      params.embeddings = embeddings
      params.offload_kqv = offload_kqv
      params.op_offload = false
      params.swa_full = true
      @handle = LibLlama.llama_init_from_model(model.to_unsafe, params)

      if @handle.null?
        error_msg = Llama.format_error(
          "Failed to create context",
          -4, # Context creation error
          "n_ctx: #{actual_n_ctx}, n_batch: #{n_batch}, n_threads: #{n_threads}, n_threads_batch: #{n_threads_batch}, embeddings: #{embeddings}, offload_kqv: #{offload_kqv}"
        )
        raise Context::Error.new(error_msg)
      end

      @model = model

      # Lazy initialization for state to avoid circular references
      @state = nil
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
      @memory ||= Memory.new(self)
    end

    # Creates a KV cache view for this context
    #
    # Parameters:
    # - n_seq_max: Maximum number of sequences per cell to track (default: 4)
    #
    # Returns:
    # - A new KvCacheView instance
    #
    # Raises:
    # - Llama::KvCacheView::Error if the view cannot be created
    def kv_cache_view(n_seq_max : Int32 = 4) : KvCacheView
      KvCacheView.new(self, n_seq_max)
    end

    # Creates a KV cache view for this context with a block
    #
    # Parameters:
    # - n_seq_max: Maximum number of sequences per cell to track (default: 4)
    # - block: The block to execute with the view
    #
    # The view will be automatically freed after the block execution
    #
    # Returns:
    # - A new KvCacheView instance
    #
    # Raises:
    # - Llama::KvCacheView::Error if the view cannot be created
    def kv_cache_view(n_seq_max : Int32 = 4, &block) : KvCacheView
      KvCacheView.new(self, n_seq_max) do |view|
        yield view
      end
    end

    # Returns the context window size (n_ctx)
    def n_ctx : UInt32
      LibLlama.llama_n_ctx(@handle)
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
    # Lazily initializes the state if it doesn't exist yet
    def state : State
      @state ||= State.new(self)
    end

    # Explicitly clean up resources
    # This can be called manually to release resources before garbage collection
    private def cleanup
      # Free the context handle
      if @handle && !@handle.null?
        LibLlama.llama_free(@handle)
      end

      # Clear references to state
      @state = nil
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

      batch = Batch.from_tokens(tokens, compute_logits_for_last, seq_ids, n_seq_max)
      decode(batch)
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
            raise TokenizationError.new(error_msg)
          end

          batch = Batch.from_tokens(tokens, true, nil, 8)
          result = decode(batch)
          results << result
        rescue ex : TokenizationError
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

    # Prepares a batch for token processing
    #
    # Parameters:
    # - all_tokens: Array of all tokens processed so far
    # - pos: Current position in the token sequence
    # - input_tokens: Original input tokens from the prompt
    #
    # Returns:
    # - A prepared Batch ready for processing
    private def prepare_batch(all_tokens : Array(Int32), pos : Int32, input_tokens : Array(Int32)) : Batch
      if pos == input_tokens.size
        # For the first iteration, process all input tokens
        # Create a batch with compute_logits_for_last=true to only compute logits for the last token
        Batch.from_tokens(input_tokens, true, nil)
      else
        # For subsequent tokens, just process the last generated token
        # Create a single-token batch with the last generated token
        last_token = all_tokens.last
        # Use Batch.from_tokens to ensure n_tokens is properly set
        batch = Batch.from_tokens([last_token], true, [0] of Int32)
        # Update the position
        batch.to_unsafe.pos[0] = pos - 1
        batch
      end
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

      # Use the internal generation method with a custom token sampler
      generate_internal(prompt, max_tokens) do |logits|
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

    # Gets the logits for the last token
    #
    # Returns:
    # - A pointer to the logits array
    def logits : Pointer(Float32)
      ptr = LibLlama.llama_get_logits(@handle)

      if ptr.null?
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
        raise Sampler::TokenizationError.new(error_msg)
      end

      # Ensure input tokens are not empty
      if input_tokens.empty?
        error_msg = Llama.format_error(
          "Tokenization resulted in empty token array",
          -6, # Tokenization error
          "prompt length: #{prompt.size}"
        )
        raise Sampler::TokenizationError.new(error_msg)
      end

      # Initialize the result string
      result = ""

      # Keep track of all tokens (input + generated)
      all_tokens = input_tokens.dup

      # Current position in the sequence
      pos = input_tokens.size

      # Generate up to max_tokens
      max_tokens.times do |i|
        begin
          # Create a batch with the current tokens
          batch = prepare_batch(all_tokens, pos, input_tokens)

          # Process the batch
          decode(batch)

          # Get the logits for the last token
          logits = self.logits

          # Sample the next token using the provided sampler
          next_token = token_sampler.call(logits)

          # Check for end of generation (EOS token)
          eos_token = @model.vocab.eos
          break if next_token == eos_token

          # Convert the token to text and add to result
          token_text = @model.vocab.token_to_text(next_token)
          result += token_text

          # Add the token to our list and increment position
          all_tokens << next_token
          pos += 1
        rescue ex : Batch::Error | TokenizationError
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

      result
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
      begin
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

          best_token
        else
          # Temperature sampling
          # Apply temperature to logits
          n_vocab = @model.vocab.n_tokens
          probs = Array(Float32).new(n_vocab, 0.0)

          # Apply temperature and convert to probabilities
          max_logit = -Float32::INFINITY
          n_vocab.times do |i|
            logits[i] /= temperature
            if logits[i] > max_logit
              max_logit = logits[i]
            end
          end

          # Compute softmax
          sum = 0.0_f32
          n_vocab.times do |i|
            probs[i] = Math.exp(logits[i] - max_logit)
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
      result = LibLlama.llama_set_adapter_lora(@handle, adapter.to_unsafe, scale)

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
      result = LibLlama.llama_rm_adapter_lora(@handle, adapter.to_unsafe)

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
      LibLlama.llama_clear_adapter_lora(@handle)
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
      result = LibLlama.llama_apply_adapter_cvec(@handle, data, data.size, n_embd, il_start, il_end)

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
    def set_embeddings(enabled : Bool)
      LibLlama.llama_set_embeddings(@handle, enabled)
    end

    # Gets the pooling type used for embeddings
    #
    # Returns:
    # - The pooling type as a PoolingType enum
    def pooling_type : LibLlama::LlamaPoolingType
      LibLlama.llama_pooling_type(@handle)
    end

    # Gets all output token embeddings
    # Only available when embeddings mode is enabled
    #
    # Returns:
    # - An array of embeddings, or nil if embeddings are not available
    #
    # Raises:
    # - Llama::Context::Error if embeddings mode is not enabled
    def get_embeddings : Array(Float32)?
      ptr = LibLlama.llama_get_embeddings(@handle)
      return nil if ptr.null?

      # Get the embedding dimension from the model
      n_embd = @model.n_embd

      # Get the number of outputs with embeddings
      # This is a simplification - in a real implementation, you would need to
      # track how many tokens had embeddings requested
      n_outputs = 1

      # Copy the embeddings to a Crystal array
      result = Array(Float32).new(n_outputs * n_embd)
      (n_outputs * n_embd).times do |i|
        result << ptr[i]
      end

      result
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
      return nil if ptr.null?

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
      return nil if ptr.null?

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
      c_messages = messages.map(&.to_unsafe)
      # First call: get required buffer size
      new_len = LibLlama.llama_chat_apply_template(
        tmpl.to_unsafe,
        c_messages.to_unsafe,
        messages.size,
        add_assistant,
        nil,
        0
      )
      raise "llama_chat_apply_template failed to get buffer size" if new_len < 0

      # Second call: allocate buffer and get the result
      buffer = Pointer(LibC::Char).malloc(new_len)
      written = LibLlama.llama_chat_apply_template(
        tmpl.to_unsafe,
        c_messages.to_unsafe,
        messages.size,
        add_assistant,
        buffer,
        new_len
      )
      raise "llama_chat_apply_template failed to write buffer" if written < 0

      String.new(buffer, written)
    end

    @handle : LibLlama::LlamaContext*
    @model : Model
    @memory : Memory?
    @state : State?

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
