module Llama
  # Wrapper for the llama_context structure
  class Context
    # Creates a new Context instance for a model
    #
    # Parameters:
    # - model: The Model to create a context for
    # - params: Optional context parameters
    #
    # Raises:
    # - Llama::Error if the context cannot be created
    def initialize(model : Model, params = nil)
      params ||= LibLlama.llama_context_default_params
      @handle = LibLlama.llama_init_from_model(model.to_unsafe, params)
      raise Error.new("Failed to create context") if @handle.null?
      @model = model
    end

    # Processes a batch of tokens
    #
    # Parameters:
    # - batch: The batch to process
    #
    # Returns:
    # - 0 on success
    # - 1 if no KV slot was found for the batch
    # - < 0 on error
    #
    # Raises:
    # - Llama::Error on error
    def decode(batch : LibLlama::LlamaBatch) : Int32
      result = LibLlama.llama_decode(@handle, batch)
      raise Error.new("Failed to decode batch") if result < 0
      result
    end

    # Gets the logits for the last token
    #
    # Returns:
    # - A pointer to the logits array
    def logits : Pointer(Float32)
      LibLlama.llama_get_logits(@handle)
    end

    # Generates text from a prompt
    #
    # Parameters:
    # - prompt: The input prompt
    # - max_tokens: Maximum number of tokens to generate
    # - temperature: Sampling temperature (0.0 = greedy, 1.0 = more random)
    #
    # Returns:
    # - The generated text
    def generate(prompt : String, max_tokens : Int32 = 128, temperature : Float32 = 0.8) : String
      # This is a simplified implementation
      # A more complete implementation would handle batching, sampling, etc.

      # Tokenize the prompt
      tokens = @model.vocab.tokenize(prompt)

      # Create a batch with the prompt tokens
      batch = LibLlama::LlamaBatch.new
      batch.n_tokens = tokens.size

      # Allocate memory for the batch
      token_ptr = Pointer(LibLlama::LlamaToken).malloc(tokens.size)
      pos_ptr = Pointer(LibLlama::LlamaPos).malloc(tokens.size)
      logits_ptr = Pointer(Int8).malloc(tokens.size)

      # Fill the batch
      tokens.each_with_index do |token, i|
        token_ptr[i] = token
        pos_ptr[i] = i.to_i32
        logits_ptr[i] = i == tokens.size - 1 ? 1_i8 : 0_i8
      end

      batch.token = token_ptr
      batch.pos = pos_ptr
      batch.logits = logits_ptr

      # Process the batch
      decode(batch)

      # Get the logits for the last token
      logits = self.logits

      # Simple greedy sampling for demonstration
      next_token = 0
      max_logit = -Float32::INFINITY

      n_vocab = @model.vocab.n_tokens
      n_vocab.times do |i|
        if logits[i] > max_logit
          max_logit = logits[i]
          next_token = i
        end
      end

      # Convert the token back to text
      @model.vocab.token_to_text(next_token)
    end

    # Returns the raw pointer to the underlying llama_context structure
    def to_unsafe
      @handle
    end

    # Frees the resources associated with this context
    def finalize
      LibLlama.llama_free(@handle) unless @handle.null?
    end

    @handle : LibLlama::LlamaContext*
    @model : Model
  end
end
