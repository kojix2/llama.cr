require "./batch/error"

module Llama
  # Wrapper for the llama_batch structure
  # Provides methods for managing batches of tokens for efficient processing
  class Batch
    # Creates a new Batch instance with the specified parameters
    #
    # Parameters:
    # - n_tokens: Maximum number of tokens this batch can hold
    # - embd: Embedding dimension (0 for token-based batch, >0 for embedding-based batch)
    # - n_seq_max: Maximum number of sequence IDs per token (default: 8)
    # Raises:
    # - ArgumentError if parameters are invalid
    # - Llama::Batch::Error if the batch cannot be created
    def initialize(n_tokens : Int32, embd : Int32 = 0, n_seq_max : Int32 = 8)
      if n_tokens <= 0
        raise ArgumentError.new("n_tokens must be positive")
      end

      if embd < 0
        raise ArgumentError.new("embd must be non-negative")
      end

      if n_seq_max <= 0
        raise ArgumentError.new("n_seq_max must be positive")
      end

      @n_seq_max = n_seq_max
      @handle = LibLlama.llama_batch_init(n_tokens, embd, n_seq_max)
      @handle.n_tokens = n_tokens

      if (embd > 0 && @handle.embd.null?) || (embd == 0 && @handle.token.null?) || @handle.pos.null? || @handle.n_seq_id.null? || @handle.seq_id.null? || @handle.logits.null?
        error_msg = Llama.format_error(
          "Failed to initialize batch",
          -2, # Memory allocation error
          "n_tokens: #{n_tokens}, embd: #{embd}"
        )
        raise Batch::Error.new(error_msg)
      end

      @owned = true
    end

    # Creates a new Batch instance from a raw llama_batch structure
    #
    # Note: This constructor is intended for internal use.
    # The batch created this way is not owned by this wrapper and will not be freed.
    def initialize(@handle : LibLlama::LlamaBatch, @owned = false, @n_seq_max : Int32 = 8)
      if @handle.n_tokens < 0
        error_msg = Llama.format_error(
          "Invalid batch handle",
          -3, # Batch processing error
          "n_tokens: #{@handle.n_tokens}"
        )
        raise Batch::Error.new(error_msg)
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
    # - Llama::Batch::Error if the batch cannot be created
    def self.get_one(tokens : Array(Int32)) : Batch
      if tokens.empty?
        # For empty token arrays, create a special batch with n_tokens=0
        # We can't use the normal constructor because it requires n_tokens > 0
        handle = LibLlama::LlamaBatch.new
        handle.n_tokens = 0
        return Batch.new(handle)
      end

      tokens_ptr = tokens.to_unsafe
      handle = LibLlama.llama_batch_get_one(tokens_ptr, tokens.size)

      if handle.n_tokens == 0
        error_msg = Llama.format_error(
          "Failed to create batch from tokens",
          -3, # Batch processing error
          "tokens size: #{tokens.size}"
        )
        raise Batch::Error.new(error_msg)
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
    # - seq_ids: Sequence IDs for all tokens (default: [0])
    # - compute_logits: Whether to compute logits for all tokens (default: true)
    #
    # Raises:
    # - ArgumentError if tokens array is empty
    # - IndexError if the batch doesn't have enough space
    # - Llama::Batch::Error if memory allocation fails
    def add_tokens(tokens : Array(Int32), pos_offset : Int32 = 0, seq_ids : Array(Int32)? = nil, compute_logits : Bool = true)
      if tokens.empty?
        raise ArgumentError.new("Tokens array cannot be empty")
      end

      if tokens.size > @handle.n_tokens
        raise IndexError.new("Batch size (#{@handle.n_tokens}) is too small for #{tokens.size} tokens")
      end

      tokens.each_with_index do |token, i|
        set_token(i, token, pos_offset + i, seq_ids, compute_logits)
      end
    end

    # Sets a token at the specified index
    #
    # Parameters:
    # - i: Index in the batch
    # - token: Token ID to set
    # - pos: Position of the token in the sequence (nil for auto-position)
    # - seq_ids: Sequence IDs (nil for default sequence 0)
    # - logits: Whether to compute logits for this token (nil for default)
    #
    # Raises:
    # - IndexError if the index is out of bounds
    # - Llama::Batch::Error if memory allocation fails
    def set_token(i : Int32, token : Int32, pos : Int32? = nil, seq_ids : Array(Int32)? = nil, logits : Bool? = nil)
      if i < 0 || i >= @handle.n_tokens
        raise IndexError.new("Index out of bounds: #{i} (valid range: 0..#{@handle.n_tokens - 1})")
      end

      # Set the token
      @handle.token[i] = token

      # Set the position
      @handle.pos[i] = pos || i

      # Set the sequence IDs
      if seq_ids.nil? || seq_ids.empty?
        @handle.n_seq_id[i] = 1
        @handle.seq_id[i][0] = 0
      else
        # Limit the number of sequence IDs to n_seq_max
        num_seq_ids = Math.min(seq_ids.size, @n_seq_max)
        @handle.n_seq_id[i] = num_seq_ids

        num_seq_ids.times do |j|
          @handle.seq_id[i][j] = seq_ids[j]
        end
      end

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
    # - seq_ids: Sequence IDs (nil for default sequence 0)
    # - logits: Whether to compute logits for this embedding (nil for default)
    #
    # Raises:
    # - IndexError if the index is out of bounds
    # - ArgumentError if the batch is not embedding-based
    # - Llama::Batch::Error if memory allocation fails
    def set_embedding(i : Int32, embedding : Array(Float32), pos : Int32? = nil, seq_ids : Array(Int32)? = nil, logits : Bool? = nil)
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

      # Set the sequence IDs
      if seq_ids.nil? || seq_ids.empty?
        @handle.n_seq_id[i] = 1
        @handle.seq_id[i][0] = 0
      else
        # Limit the number of sequence IDs to n_seq_max
        num_seq_ids = Math.min(seq_ids.size, @n_seq_max)
        @handle.n_seq_id[i] = num_seq_ids

        num_seq_ids.times do |j|
          @handle.seq_id[i][j] = seq_ids[j]
        end
      end

      # Set the logits flag if provided
      if logits
        @handle.logits[i] = logits ? 1_i8 : 0_i8
      end
    end

    # Factory methods for common batch creation patterns

    # Crystal implementation of llama_batch_get_one that properly allocates memory
    # This is used instead of the C function which doesn't allocate memory for pos, seq_id, etc.
    private def self.crystal_llama_batch_get_one(tokens : Pointer(Int32), n : Int32, n_seq_max : Int32 = 8) : Tuple(LibLlama::LlamaBatch, Bool)
      batch = LibLlama.llama_batch_init(n, 0, n_seq_max)

      # Allocate new memory for tokens and copy the data
      new_tokens = Pointer(Int32).malloc(n)
      new_tokens.copy_from(tokens, n)

      batch.token = new_tokens
      batch.n_tokens = n

      # Return the batch and a flag indicating that it has Crystal-allocated token memory
      {batch, true}
    end

    # Creates a batch for a sequence of tokens with optional parameters
    #
    # Parameters:
    # - tokens: Array of token IDs
    # - compute_logits_for_last: Whether to compute logits only for the last token
    # - seq_ids: Sequence IDs to use for all tokens (default: nil)
    # - n_seq_max: Maximum number of sequence IDs per token (default: 8)
    #
    # Returns:
    # - A new Batch instance configured with the provided tokens
    #
    # Raises:
    # - ArgumentError if tokens array is empty
    # - Llama::Batch::Error if batch creation fails
    def self.from_tokens(tokens : Array(Int32), compute_logits_for_last : Bool = true, seq_ids : Array(Int32)? = nil, n_seq_max : Int32 = 8) : Batch
      if tokens.empty?
        raise ArgumentError.new("Tokens array cannot be empty")
      end

      begin
        # Use custom function to create a batch with memory allocated
        handle, has_crystal_token = self.crystal_llama_batch_get_one(tokens.to_unsafe, tokens.size, n_seq_max)
        # Explicitly set owned=true since we created this batch and need to free it
        batch = Batch.new(handle, owned: true)
        # Set the flag indicating that this batch has Crystal-allocated token memory
        batch.has_crystal_token = has_crystal_token

        # Set position, sequence ID, and logits flag for each token
        tokens.size.times do |i|
          # Set the position
          batch.to_unsafe.pos[i] = i

          # Set the sequence IDs
          if seq_ids.nil? || seq_ids.empty?
            batch.to_unsafe.n_seq_id[i] = 1
            batch.to_unsafe.seq_id[i][0] = 0
          else
            # Limit the number of sequence IDs to n_seq_max
            num_seq_ids = Math.min(seq_ids.size, n_seq_max)
            batch.to_unsafe.n_seq_id[i] = num_seq_ids

            num_seq_ids.times do |j|
              batch.to_unsafe.seq_id[i][j] = seq_ids[j]
            end
          end

          # Set the logits flag
          needs_logits = compute_logits_for_last ? (i == tokens.size - 1) : true
          batch.to_unsafe.logits[i] = needs_logits ? 1_i8 : 0_i8
        end

        batch
      rescue ex : Batch::Error | ArgumentError | IndexError
        raise ex
      rescue ex
        error_msg = Llama.format_error(
          "Failed to create batch for tokens",
          -3, # Batch processing error
          "tokens size: #{tokens.size}, error: #{ex.message}"
        )
        raise Batch::Error.new(error_msg)
      end
    end

    # Creates a batch for embeddings with optional parameters
    #
    # Parameters:
    # - embeddings: Array of embedding vectors
    # - seq_ids: Sequence IDs to use for all embeddings (default: nil)
    # - n_seq_max: Maximum number of sequence IDs per token (default: 8)
    #
    # Returns:
    # - A new Batch instance configured with the provided embeddings
    #
    # Raises:
    # - ArgumentError if embeddings array is empty or contains empty embeddings
    # - Llama::Batch::Error if batch creation fails
    def self.for_embeddings(embeddings : Array(Array(Float32)), seq_ids : Array(Int32)? = nil, n_seq_max : Int32 = 8) : Batch
      if embeddings.empty?
        raise ArgumentError.new("Embeddings array cannot be empty")
      end

      if embeddings.first.empty?
        raise ArgumentError.new("Embedding vectors cannot be empty")
      end

      begin
        embd_size = embeddings.first.size
        batch = Batch.new(embeddings.size, embd_size, n_seq_max)

        embeddings.each_with_index do |embedding, i|
          if embedding.size != embd_size
            error_msg = Llama.format_error(
              "Inconsistent embedding dimensions",
              nil,
              "expected: #{embd_size}, got: #{embedding.size} at index #{i}"
            )
            raise Batch::Error.new(error_msg)
          end

          batch.set_embedding(i, embedding, i, seq_ids)
        end

        batch
      rescue ex : Batch::Error | ArgumentError | IndexError
        raise ex
      rescue ex
        error_msg = Llama.format_error(
          "Failed to create batch for embeddings",
          -3, # Batch processing error
          "embeddings size: #{embeddings.size}, embd_size: #{embeddings.first.size}, error: #{ex.message}"
        )
        raise Batch::Error.new(error_msg)
      end
    end

    # Returns the raw pointer to the underlying llama_batch structure
    def to_unsafe
      @handle
    end

    # Setter for has_crystal_token
    def has_crystal_token=(value : Bool)
      @has_crystal_token = value
    end

    # Explicitly clean up resources
    # This can be called manually to release resources before garbage collection
    private def cleanup
      if @owned
        LibLlama.llama_batch_free(to_unsafe)
        @owned = false
      end
    end

    # Frees the resources associated with this batch
    def finalize
      cleanup
    end

    @handle : LibLlama::LlamaBatch
    @owned : Bool
    @has_crystal_token : Bool = false # Flag to indicate if token memory was allocated by Crystal

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
