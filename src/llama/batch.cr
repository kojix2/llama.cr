module Llama
  # Wrapper for the llama_batch structure
  # Provides methods for managing batches of tokens for efficient processing
  class Batch
    # Creates a new Batch instance with the specified parameters
    #
    # Parameters:
    # - n_tokens: Maximum number of tokens this batch can hold
    # - embd: Embedding dimension (0 for token-based batch, >0 for embedding-based batch)
    # - n_seq_max: Maximum number of sequences per token
    #
    # Raises:
    # - ArgumentError if parameters are invalid
    # - Llama::BatchError if the batch cannot be created
    def initialize(n_tokens : Int32, embd : Int32 = 0, n_seq_max : Int32 = 1)
      if n_tokens <= 0
        raise ArgumentError.new("n_tokens must be positive")
      end

      if embd < 0
        raise ArgumentError.new("embd must be non-negative")
      end

      if n_seq_max <= 0
        raise ArgumentError.new("n_seq_max must be positive")
      end

      @handle = LibLlama.llama_batch_init(n_tokens, embd, n_seq_max)

      # Check if batch initialization failed
      if @handle.n_tokens == 0
        # Initialize fields to avoid null pointer exceptions
        @handle.token = Pointer(LibLlama::LlamaToken).null
        @handle.embd = Pointer(Float32).null
        @handle.pos = Pointer(LibLlama::LlamaPos).null
        @handle.n_seq_id = Pointer(Int32).null
        @handle.seq_id = Pointer(Pointer(LibLlama::LlamaSeqId)).null
        @handle.logits = Pointer(Int8).null

        # Manually allocate memory for the batch
        @handle.n_tokens = n_tokens

        begin
          @handle.token = Pointer(LibLlama::LlamaToken).malloc(n_tokens)
          @handle.pos = Pointer(LibLlama::LlamaPos).malloc(n_tokens)

          if embd > 0
            @handle.embd = Pointer(Float32).malloc(n_tokens * embd)
          end
        rescue ex
          error_msg = Llama.format_error(
            "Failed to allocate memory for batch",
            -2, # Memory allocation error
            "n_tokens: #{n_tokens}, embd: #{embd}, n_seq_max: #{n_seq_max}"
          )
          raise BatchError.new(error_msg)
        end

        @owned = true
      else
        @owned = true
      end
    end

    # Creates a new Batch instance from a raw llama_batch structure
    #
    # Note: This constructor is intended for internal use.
    # The batch created this way is not owned by this wrapper and will not be freed.
    def initialize(@handle : LibLlama::LlamaBatch, @owned = false)
      if @handle.n_tokens < 0
        error_msg = Llama.format_error(
          "Invalid batch handle",
          -3, # Batch processing error
          "n_tokens: #{@handle.n_tokens}"
        )
        raise BatchError.new(error_msg)
      end
    end

    # Creates a new Batch for a single sequence of tokens
    #
    # Parameters:
    # - tokens: Array of token IDs
    #
    # Returns:
    # - A new Batch instance
    #
    # Raises:
    # - Llama::BatchError if the batch cannot be created
    def self.get_one(tokens : Array(Int32)) : Batch
      if tokens.empty?
        error_msg = Llama.format_error(
          "Cannot create batch from empty token array",
          -3, # Batch processing error
          nil
        )
        raise BatchError.new(error_msg)
      end

      tokens_ptr = tokens.to_unsafe
      handle = LibLlama.llama_batch_get_one(tokens_ptr, tokens.size)

      if handle.n_tokens == 0
        error_msg = Llama.format_error(
          "Failed to create batch from tokens",
          -3, # Batch processing error
          "tokens size: #{tokens.size}"
        )
        raise BatchError.new(error_msg)
      end

      Batch.new(handle)
    end

    # Returns the number of tokens in this batch
    def n_tokens : Int32
      @handle.n_tokens
    end

    # Internal helper methods for memory management

    # Ensures the token buffer is allocated
    private def ensure_token_buffer
      if @handle.token.null?
        begin
          @handle.token = Pointer(LibLlama::LlamaToken).malloc(@handle.n_tokens)
        rescue ex
          error_msg = Llama.format_error(
            "Failed to allocate token buffer",
            -2, # Memory allocation error
            "n_tokens: #{@handle.n_tokens}"
          )
          raise BatchError.new(error_msg)
        end
      end
    end

    # Ensures the position buffer is allocated
    private def ensure_pos_buffer
      if @handle.pos.null?
        begin
          @handle.pos = Pointer(LibLlama::LlamaPos).malloc(@handle.n_tokens)
        rescue ex
          error_msg = Llama.format_error(
            "Failed to allocate position buffer",
            -2, # Memory allocation error
            "n_tokens: #{@handle.n_tokens}"
          )
          raise BatchError.new(error_msg)
        end
      end
    end

    # Ensures the sequence ID buffers are allocated
    private def ensure_seq_id_buffer(i : Int32)
      if @handle.n_seq_id.null?
        begin
          @handle.n_seq_id = Pointer(Int32).malloc(@handle.n_tokens)
          @handle.n_tokens.times do |j|
            @handle.n_seq_id[j] = 0
          end
        rescue ex
          error_msg = Llama.format_error(
            "Failed to allocate n_seq_id buffer",
            -2, # Memory allocation error
            "n_tokens: #{@handle.n_tokens}"
          )
          raise BatchError.new(error_msg)
        end
      end

      if @handle.seq_id.null?
        begin
          @handle.seq_id = Pointer(Pointer(Int32)).malloc(@handle.n_tokens)
          @handle.n_tokens.times do |j|
            @handle.seq_id[j] = Pointer(Int32).null
          end
        rescue ex
          error_msg = Llama.format_error(
            "Failed to allocate seq_id buffer",
            -2, # Memory allocation error
            "n_tokens: #{@handle.n_tokens}"
          )
          raise BatchError.new(error_msg)
        end
      end

      if @handle.seq_id[i].null?
        begin
          @handle.seq_id[i] = Pointer(Int32).malloc(1)
        rescue ex
          error_msg = Llama.format_error(
            "Failed to allocate seq_id[#{i}] buffer",
            -2, # Memory allocation error
            nil
          )
          raise BatchError.new(error_msg)
        end
      end
    end

    # Ensures the logits buffer is allocated
    private def ensure_logits_buffer
      if @handle.logits.null?
        begin
          @handle.logits = Pointer(Int8).malloc(@handle.n_tokens)
          @handle.n_tokens.times do |j|
            @handle.logits[j] = 0_i8
          end
        rescue ex
          error_msg = Llama.format_error(
            "Failed to allocate logits buffer",
            -2, # Memory allocation error
            "n_tokens: #{@handle.n_tokens}"
          )
          raise BatchError.new(error_msg)
        end
      end
    end

    # Adds multiple tokens to the batch
    #
    # Parameters:
    # - tokens: Array of token IDs to add
    # - pos_offset: Position offset for the tokens (default: 0)
    # - seq_id: Sequence ID for all tokens (default: 0)
    # - compute_logits: Whether to compute logits for all tokens (default: true)
    #
    # Raises:
    # - ArgumentError if tokens array is empty
    # - IndexError if the batch doesn't have enough space
    # - Llama::BatchError if memory allocation fails
    def add_tokens(tokens : Array(Int32), pos_offset : Int32 = 0, seq_id : Int32 = 0, compute_logits : Bool = true)
      if tokens.empty?
        raise ArgumentError.new("Tokens array cannot be empty")
      end

      if tokens.size > @handle.n_tokens
        raise IndexError.new("Batch size (#{@handle.n_tokens}) is too small for #{tokens.size} tokens")
      end

      tokens.each_with_index do |token, i|
        set_token(i, token, pos_offset + i, seq_id, compute_logits)
      end
    end

    # Sets a token at the specified index
    #
    # Parameters:
    # - i: Index in the batch
    # - token: Token ID to set
    # - pos: Position of the token in the sequence (nil for auto-position)
    # - seq_id: Sequence ID (nil for default sequence 0)
    # - logits: Whether to compute logits for this token (nil for default)
    #
    # Raises:
    # - IndexError if the index is out of bounds
    # - Llama::BatchError if memory allocation fails
    def set_token(i : Int32, token : Int32, pos : Int32? = nil, seq_id : Int32? = nil, logits : Bool? = nil)
      if i < 0 || i >= @handle.n_tokens
        raise IndexError.new("Index out of bounds: #{i} (valid range: 0..#{@handle.n_tokens - 1})")
      end

      # Ensure buffers are allocated
      begin
        ensure_token_buffer
        ensure_pos_buffer
        ensure_seq_id_buffer(i)
      rescue ex : BatchError
        raise ex
      rescue ex
        error_msg = Llama.format_error(
          "Failed to ensure buffers for set_token",
          -2, # Memory allocation error
          "index: #{i}, error: #{ex.message}"
        )
        raise BatchError.new(error_msg)
      end

      # Set the token
      @handle.token[i] = token

      # Set the position
      @handle.pos[i] = pos || i

      # Set the sequence ID
      @handle.n_seq_id[i] = 1
      @handle.seq_id[i][0] = seq_id || 0

      # Set the logits flag if provided
      if logits
        ensure_logits_buffer
        @handle.logits[i] = logits ? 1_i8 : 0_i8
      end
    end

    # Ensures the embedding buffer is allocated
    private def ensure_embd_buffer(embd_size : Int32)
      if @handle.embd.null?
        begin
          @handle.embd = Pointer(Float32).malloc(@handle.n_tokens * embd_size)
        rescue ex
          error_msg = Llama.format_error(
            "Failed to allocate embedding buffer",
            -2, # Memory allocation error
            "n_tokens: #{@handle.n_tokens}, embd_size: #{embd_size}"
          )
          raise BatchError.new(error_msg)
        end
      end
    end

    # Sets an embedding at the specified index
    #
    # Parameters:
    # - i: Index in the batch
    # - embedding: Array of embedding values
    # - pos: Position of the embedding in the sequence (nil for auto-position)
    # - seq_id: Sequence ID (nil for default sequence 0)
    # - logits: Whether to compute logits for this embedding (nil for default)
    #
    # Raises:
    # - IndexError if the index is out of bounds
    # - ArgumentError if the batch is not embedding-based
    # - Llama::BatchError if memory allocation fails
    def set_embedding(i : Int32, embedding : Array(Float32), pos : Int32? = nil, seq_id : Int32? = nil, logits : Bool? = nil)
      if i < 0 || i >= @handle.n_tokens
        raise IndexError.new("Index out of bounds: #{i} (valid range: 0..#{@handle.n_tokens - 1})")
      end

      if embedding.empty?
        raise ArgumentError.new("Embedding array cannot be empty")
      end

      # Ensure buffers are allocated
      embd_size = embedding.size
      begin
        ensure_embd_buffer(embd_size)
        ensure_pos_buffer
        ensure_seq_id_buffer(i)
      rescue ex : BatchError
        raise ex
      rescue ex
        error_msg = Llama.format_error(
          "Failed to ensure buffers for set_embedding",
          -2, # Memory allocation error
          "index: #{i}, embd_size: #{embd_size}, error: #{ex.message}"
        )
        raise BatchError.new(error_msg)
      end

      # Copy the embedding values
      embd_size.times do |j|
        @handle.embd[i * embd_size + j] = embedding[j]
      end

      # Set the position
      @handle.pos[i] = pos || i

      # Set the sequence ID
      @handle.n_seq_id[i] = 1
      @handle.seq_id[i][0] = seq_id || 0

      # Set the logits flag if provided
      if logits
        ensure_logits_buffer
        @handle.logits[i] = logits ? 1_i8 : 0_i8
      end
    end

    # Factory methods for common batch creation patterns

    # Creates a batch for a sequence of tokens with optional parameters
    #
    # Parameters:
    # - tokens: Array of token IDs
    # - compute_logits_for_last: Whether to compute logits only for the last token
    # - seq_id: Sequence ID to use for all tokens
    #
    # Returns:
    # - A new Batch instance configured with the provided tokens
    #
    # Raises:
    # - ArgumentError if tokens array is empty
    # - Llama::BatchError if batch creation fails
    def self.for_tokens(tokens : Array(Int32), compute_logits_for_last : Bool = true, seq_id : Int32 = 0) : Batch
      if tokens.empty?
        raise ArgumentError.new("Tokens array cannot be empty")
      end

      begin
        batch = Batch.new(tokens.size)

        tokens.each_with_index do |token, i|
          # Determine if we need logits for this token
          needs_logits = compute_logits_for_last ? (i == tokens.size - 1) : true
          batch.set_token(i, token, i, seq_id, needs_logits)
        end

        batch
      rescue ex : BatchError | ArgumentError | IndexError
        raise ex
      rescue ex
        error_msg = Llama.format_error(
          "Failed to create batch for tokens",
          -3, # Batch processing error
          "tokens size: #{tokens.size}, error: #{ex.message}"
        )
        raise BatchError.new(error_msg)
      end
    end

    # Creates a batch for embeddings with optional parameters
    #
    # Parameters:
    # - embeddings: Array of embedding vectors
    # - seq_id: Sequence ID to use for all embeddings
    #
    # Returns:
    # - A new Batch instance configured with the provided embeddings
    #
    # Raises:
    # - ArgumentError if embeddings array is empty or contains empty embeddings
    # - Llama::BatchError if batch creation fails
    def self.for_embeddings(embeddings : Array(Array(Float32)), seq_id : Int32 = 0) : Batch
      if embeddings.empty?
        raise ArgumentError.new("Embeddings array cannot be empty")
      end

      if embeddings.first.empty?
        raise ArgumentError.new("Embedding vectors cannot be empty")
      end

      begin
        embd_size = embeddings.first.size
        batch = Batch.new(embeddings.size, embd_size)

        embeddings.each_with_index do |embedding, i|
          if embedding.size != embd_size
            error_msg = Llama.format_error(
              "Inconsistent embedding dimensions",
              nil,
              "expected: #{embd_size}, got: #{embedding.size} at index #{i}"
            )
            raise BatchError.new(error_msg)
          end

          batch.set_embedding(i, embedding, i, seq_id)
        end

        batch
      rescue ex : BatchError | ArgumentError | IndexError
        raise ex
      rescue ex
        error_msg = Llama.format_error(
          "Failed to create batch for embeddings",
          -3, # Batch processing error
          "embeddings size: #{embeddings.size}, embd_size: #{embeddings.first.size}, error: #{ex.message}"
        )
        raise BatchError.new(error_msg)
      end
    end

    # Returns the raw pointer to the underlying llama_batch structure
    def to_unsafe
      @handle
    end

    # Explicitly clean up resources
    # This can be called manually to release resources before garbage collection
    def cleanup
      if @owned && @handle
        LibLlama.llama_batch_free(@handle)
        @handle = LibLlama::LlamaBatch.new
      end
    end

    # Frees the resources associated with this batch
    def finalize
      cleanup
    end

    @handle : LibLlama::LlamaBatch
    @owned : Bool
  end
end
