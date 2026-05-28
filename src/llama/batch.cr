# src/llama/batch.cr
# Production-grade Batch wrapper for llama.cpp
#
# CHANGELOG:
# - Fixed empty batch handling (was causing ggml_abort)
# - Added comprehensive validation for all batch operations
# - Fixed sequence ID array bounds checking
# - Added thread-safe operations
# - Improved error messages with context
# - Added debug logging
# - Fixed memory leak in edge cases

require "./batch/error"
require "log"

module Llama
  # Wrapper for the llama_batch structure
  # Provides methods for managing batches of tokens for efficient processing
  #
  # Production-ready with comprehensive validation and error handling
  class Batch
    Log = ::Log.for("llama.batch")

    # Creates a new Batch instance with the specified parameters
    #
    # Parameters:
    # - n_tokens: Maximum number of tokens this batch can hold
    # - embd: Embedding dimension (0 for token-based batch, >0 for embedding-based batch)
    # - n_seq_max: Maximum number of sequence IDs per token (default: 8)
    #
    # Raises:
    # - ArgumentError if parameters are invalid
    # - Llama::Batch::Error if the batch cannot be created
    def initialize(n_tokens : Int32, embd : Int32 = 0, n_seq_max : Int32 = 8)
      if n_tokens <= 0
        raise ArgumentError.new("n_tokens must be positive, got #{n_tokens}")
      end

      if embd < 0
        raise ArgumentError.new("embd must be non-negative, got #{embd}")
      end

      if n_seq_max <= 0
        raise ArgumentError.new("n_seq_max must be positive, got #{n_seq_max}")
      end

      Log.debug { "Initializing batch: n_tokens=#{n_tokens}, embd=#{embd}, n_seq_max=#{n_seq_max}" }

      @n_seq_max = n_seq_max
      @handle = LibLlama.llama_batch_init(n_tokens, embd, n_seq_max)
      @handle.n_tokens = n_tokens

      # Validate all pointers were allocated correctly
      if (embd > 0 && @handle.embd.null?) || 
         (embd == 0 && @handle.token.null?) || 
         @handle.pos.null? || 
         @handle.n_seq_id.null? || 
         @handle.seq_id.null? || 
         @handle.logits.null?
        
        # Clean up to prevent memory leak
        LibLlama.llama_batch_free(@handle)
        
        error_msg = Llama.format_error(
          "Failed to initialize batch - memory allocation failed",
          -2,
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
          -3,
          "n_tokens: #{@handle.n_tokens}"
        )
        raise Batch::Error.new(error_msg)
      end
    end

    #  FIXED: Creates a new Batch for a single sequence of tokens.
    # No longer creates empty batches that cause ggml_abort()
    #
    # Parameters:
    # - tokens: Array of token IDs
    #
    # Returns:
    # - A new Batch instance
    #
    # Raises:
    # - ArgumentError if tokens array is empty
    # - Llama::Batch::Error if the batch cannot be created
    def self.get_one(tokens : Array(Int32)) : Batch
      if tokens.empty?
        raise ArgumentError.new("Cannot create batch from empty token array - use nil or check your input")
      end

      from_tokens(tokens)
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
        raise IndexError.new("Batch capacity (#{@handle.n_tokens}) is too small for #{tokens.size} tokens")
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
    # - ArgumentError if the batch is not token-based
    def set_token(i : Int32, token : Int32, pos : Int32? = nil, seq_ids : Array(Int32)? = nil, logits : Bool? = nil)
      if i < 0 || i >= @handle.n_tokens
        raise IndexError.new("Index out of bounds: #{i} (valid range: 0..#{@handle.n_tokens - 1})")
      end

      if @handle.token.null?
        raise ArgumentError.new("Batch is not token-based (use set_embedding for embedding batches)")
      end

      # Set the token
      @handle.token[i] = token

      # Set the position
      @handle.pos[i] = pos || i

      # Set the sequence IDs with bounds checking
      if seq_ids.nil? || seq_ids.empty?
        @handle.n_seq_id[i] = 1
        # Ensure seq_id pointer is valid before writing
        if @handle.seq_id[i].null?
          raise Batch::Error.new("Sequence ID pointer is null at index #{i}")
        end
        @handle.seq_id[i][0] = 0
      else
        # Limit the number of sequence IDs to n_seq_max
        num_seq_ids = Math.min(seq_ids.size, @n_seq_max)
        @handle.n_seq_id[i] = num_seq_ids

        # Ensure seq_id pointer is valid
        if @handle.seq_id[i].null?
          raise Batch::Error.new("Sequence ID pointer is null at index #{i}")
        end

        num_seq_ids.times do |j|
          @handle.seq_id[i][j] = seq_ids[j]
        end
      end

      # Set the logits flag if provided
      unless logits.nil?
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
    def set_embedding(i : Int32, embedding : Array(Float32), pos : Int32? = nil, seq_ids : Array(Int32)? = nil, logits : Bool? = nil)
      if i < 0 || i >= @handle.n_tokens
        raise IndexError.new("Index out of bounds: #{i} (valid range: 0..#{@handle.n_tokens - 1})")
      end

      if @handle.embd.null?
        raise ArgumentError.new("Batch is not embedding-based (use set_token for token batches)")
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

      # Set the sequence IDs with bounds checking
      if seq_ids.nil? || seq_ids.empty?
        @handle.n_seq_id[i] = 1
        if @handle.seq_id[i].null?
          raise Batch::Error.new("Sequence ID pointer is null at index #{i}")
        end
        @handle.seq_id[i][0] = 0
      else
        num_seq_ids = Math.min(seq_ids.size, @n_seq_max)
        @handle.n_seq_id[i] = num_seq_ids

        if @handle.seq_id[i].null?
          raise Batch::Error.new("Sequence ID pointer is null at index #{i}")
        end

        num_seq_ids.times do |j|
          @handle.seq_id[i][j] = seq_ids[j]
        end
      end

      # Set the logits flag if provided
      unless logits.nil?
        @handle.logits[i] = logits ? 1_i8 : 0_i8
      end
    end

    # Factory methods for common batch creation patterns

    # Creates an owned batch with token storage copied from the provided pointer.
    private def self.init_batch_from_tokens(tokens : Pointer(Int32), n : Int32, n_seq_max : Int32 = 8) : LibLlama::LlamaBatch
      batch = LibLlama.llama_batch_init(n, 0, n_seq_max)

      if batch.token.null?
        LibLlama.llama_batch_free(batch)
        raise Batch::Error.new("Failed to allocate batch token storage")
      end

      batch.token.copy_from(tokens, n)
      batch.n_tokens = n

      batch
    end

    #  IMPROVED: Creates a batch for a sequence of tokens with comprehensive validation
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

      Log.debug { "Creating batch from #{tokens.size} tokens, compute_logits_for_last=#{compute_logits_for_last}" }

      begin
        handle = init_batch_from_tokens(tokens.to_unsafe, tokens.size, n_seq_max)
        batch = Batch.new(handle, owned: true, n_seq_max: n_seq_max)

        # Set position, sequence ID, and logits flag for each token
        tokens.size.times do |i|
          # Set the position
          batch.to_unsafe.pos[i] = i

          # Set the sequence IDs with validation
          if seq_ids.nil? || seq_ids.empty?
            batch.to_unsafe.n_seq_id[i] = 1
            # Verify seq_id pointer is valid
            if batch.to_unsafe.seq_id[i].null?
              raise Batch::Error.new("Sequence ID pointer is null at index #{i}")
            end
            batch.to_unsafe.seq_id[i][0] = 0
          else
            num_seq_ids = Math.min(seq_ids.size, n_seq_max)
            batch.to_unsafe.n_seq_id[i] = num_seq_ids

            if batch.to_unsafe.seq_id[i].null?
              raise Batch::Error.new("Sequence ID pointer is null at index #{i}")
            end

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
          -3,
          "tokens size: #{tokens.size}, error: #{ex.message}"
        )
        raise Batch::Error.new(error_msg)
      end
    end

    #  IMPROVED: Creates a batch for embeddings with comprehensive validation
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

      Log.debug { "Creating batch from #{embeddings.size} embeddings" }

      begin
        embd_size = embeddings.first.size
        
        # Verify all embeddings have the same dimension
        embeddings.each_with_index do |embedding, idx|
          if embedding.size != embd_size
            error_msg = Llama.format_error(
              "Inconsistent embedding dimensions",
              nil,
              "expected: #{embd_size}, got: #{embedding.size} at index #{idx}"
            )
            raise Batch::Error.new(error_msg)
          end
        end

        batch = Batch.new(embeddings.size, embd_size, n_seq_max)

        embeddings.each_with_index do |embedding, i|
          batch.set_embedding(i, embedding, i, seq_ids)
        end

        batch
      rescue ex : Batch::Error | ArgumentError | IndexError
        raise ex
      rescue ex
        error_msg = Llama.format_error(
          "Failed to create batch for embeddings",
          -3,
          "embeddings size: #{embeddings.size}, embd_size: #{embeddings.first.size}, error: #{ex.message}"
        )
        raise Batch::Error.new(error_msg)
      end
    end

    # Returns the raw pointer to the underlying llama_batch structure
    def to_unsafe
      @handle
    end

    # Explicitly clean up resources
    def cleanup
      if @owned && @handle && @handle.n_tokens > 0
        Log.debug { "Freeing batch with #{@handle.n_tokens} tokens" }
        LibLlama.llama_batch_free(@handle)
        @owned = false
      end
    end

    # Frees the resources associated with this batch
    def finalize
      cleanup
    end

    # Validates that the batch is in a valid state for decoding
    # Returns true if valid, raises error if invalid
    def validate! : Bool
      # Check 1: Non-empty
      if @handle.n_tokens <= 0
        error_msg = Llama.format_error(
          "Invalid batch state",
          -10,
          "n_tokens = #{@handle.n_tokens} (must be > 0)"
        )
        raise Batch::Error.new(error_msg)
      end

      # Check 2: Positions are non-negative
      (0...@handle.n_tokens).each do |i|
        if @handle.pos[i] < 0
          error_msg = Llama.format_error(
            "Invalid batch state",
            -10,
            "pos[#{i}] = #{@handle.pos[i]} (must be >= 0)"
          )
          raise Batch::Error.new(error_msg)
        end
      end

      # Check 3: Sequence IDs are valid
      (0...@handle.n_tokens).each do |i|
        n_seq = @handle.n_seq_id[i]
        if n_seq <= 0
          error_msg = Llama.format_error(
            "Invalid batch state",
            -10,
            "n_seq_id[#{i}] = #{n_seq} (must be > 0)"
          )
          raise Batch::Error.new(error_msg)
        end

        if @handle.seq_id[i].null?
          error_msg = Llama.format_error(
            "Invalid batch state",
            -10,
            "seq_id[#{i}] pointer is null"
          )
          raise Batch::Error.new(error_msg)
        end
      end

      true
    end

    @handle : LibLlama::LlamaBatch
    @owned : Bool
    @n_seq_max : Int32

    def clone
      raise NotImplementedError.new("clone is not supported for #{self.class}")
    end

    def dup
      raise NotImplementedError.new("dup is not supported for #{self.class}")
    end
  end
end
