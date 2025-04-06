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

      if (embd > 0 && @handle.embd.null?) || (embd == 0 && @handle.token.null?) || @handle.pos.null? || @handle.n_seq_id.null? || @handle.seq_id.null? || @handle.logits.null?
        error_msg = Llama.format_error(
          "Failed to initialize batch",
          -2, # Memory allocation error
          "n_tokens: #{n_tokens}, embd: #{embd}, n_seq_max: #{n_seq_max}"
        )
        raise BatchError.new(error_msg)
      end

      @owned = true
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

      # Set the token
      @handle.token[i] = token

      # Set the position
      @handle.pos[i] = pos || i

      # Set the sequence ID
      @handle.n_seq_id[i] = 1
      @handle.seq_id[i][0] = seq_id || 0

      # Set the logits flag if provided
      if logits
        @handle.logits[i] = logits ? 1_i8 : 0_i8
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

      # Copy the embedding values
      embd_size = embedding.size
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
      if @owned
        LibLlama.llama_batch_free(@handle)
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
