require "./kv_cache/error"

module Llama
  # Wrapper for the llama_kv_cache structure
  # Provides methods for managing the KV (Key-Value) cache in LLaMA models
  class KvCache
    # Creates a new KvCache instance from a raw pointer
    #
    # Note: This constructor is intended for internal use.
    # Users should obtain KvCache instances through Context#kv_cache.
    #
    # To avoid circular references, we store the context pointer rather than the context object
    #
    # Raises:
    # - Llama::KvCache::Error if the handle is null
    def initialize(@handle : LibLlama::LlamaKvCache*, ctx : Context)
      if @handle.null?
        error_msg = Llama.format_error(
          "Failed to initialize KV cache",
          -7, # KV cache error
          "handle is null"
        )
        raise KvCache::Error.new(error_msg)
      end

      @ctx_ptr = ctx.to_unsafe

      if @ctx_ptr.null?
        error_msg = Llama.format_error(
          "Failed to initialize KV cache",
          -7, # KV cache error
          "context pointer is null"
        )
        raise KvCache::Error.new(error_msg)
      end
    end

    # Get the context pointer for internal use
    private def ctx_ptr : LibLlama::LlamaContext*
      if @ctx_ptr.null?
        error_msg = Llama.format_error(
          "Invalid context pointer",
          -7, # KV cache error
          "context pointer is null"
        )
        raise KvCache::Error.new(error_msg)
      end

      @ctx_ptr
    end

    # Clears the KV cache
    # This removes all tokens from the cache and resets its state
    #
    # Returns:
    # - self for method chaining
    #
    # Raises:
    # - Llama::KvCache::Error if the operation fails
    def clear
      begin
        LibLlama.llama_kv_self_clear(ctx_ptr)
        self
      rescue ex
        error_msg = Llama.format_error(
          "Failed to clear KV cache",
          -7, # KV cache error
          ex.message
        )
        raise KvCache::Error.new(error_msg)
      end
    end

    # Returns the number of tokens in the KV cache
    # If a KV cell has multiple sequences assigned to it, it will be counted multiple times
    #
    # Returns:
    # - The number of tokens in the KV cache
    #
    # Raises:
    # - Llama::KvCache::Error if the operation fails
    def n_tokens : Int32
      begin
        LibLlama.llama_kv_self_n_tokens(ctx_ptr)
      rescue ex
        error_msg = Llama.format_error(
          "Failed to get number of tokens in KV cache",
          -7, # KV cache error
          ex.message
        )
        raise KvCache::Error.new(error_msg)
      end
    end

    # Returns the number of used KV cells
    # A cell is considered used if it has at least one sequence assigned to it
    #
    # Returns:
    # - The number of used KV cells
    #
    # Raises:
    # - Llama::KvCache::Error if the operation fails
    def used_cells : Int32
      begin
        LibLlama.llama_kv_self_used_cells(ctx_ptr)
      rescue ex
        error_msg = Llama.format_error(
          "Failed to get number of used cells in KV cache",
          -7, # KV cache error
          ex.message
        )
        raise KvCache::Error.new(error_msg)
      end
    end

    # Removes tokens from a sequence in the KV cache
    #
    # Parameters:
    # - seq_id: The sequence ID to remove tokens from (seq_id < 0 matches any sequence)
    # - p0: Start position (p0 < 0 means start from 0)
    # - p1: End position (p1 < 0 means end at infinity)
    #
    # Returns:
    # - true if successful, false if a partial sequence cannot be removed
    #   (removing a whole sequence never fails)
    #
    # Raises:
    # - Llama::KvCache::Error if the operation fails
    def seq_rm(seq_id : Int32, p0 : Int32, p1 : Int32) : Bool
      begin
        result = LibLlama.llama_kv_self_seq_rm(ctx_ptr, seq_id, p0, p1)

        # This is not an error, just a result indicating that a partial sequence cannot be removed
        return result
      rescue ex
        error_msg = Llama.format_error(
          "Failed to remove sequence from KV cache",
          -7, # KV cache error
          "seq_id: #{seq_id}, p0: #{p0}, p1: #{p1}, error: #{ex.message}"
        )
        raise KvCache::Error.new(error_msg)
      end
    end

    # Copies tokens from one sequence to another in the KV cache
    #
    # Parameters:
    # - seq_id_src: Source sequence ID
    # - seq_id_dst: Destination sequence ID
    # - p0: Start position (p0 < 0 means start from 0)
    # - p1: End position (p1 < 0 means end at infinity)
    #
    # Returns:
    # - self for method chaining
    #
    # Raises:
    # - Llama::KvCache::Error if the operation fails
    def seq_cp(seq_id_src : Int32, seq_id_dst : Int32, p0 : Int32, p1 : Int32)
      begin
        LibLlama.llama_kv_self_seq_cp(ctx_ptr, seq_id_src, seq_id_dst, p0, p1)
        self
      rescue ex
        error_msg = Llama.format_error(
          "Failed to copy sequence in KV cache",
          -7, # KV cache error
          "seq_id_src: #{seq_id_src}, seq_id_dst: #{seq_id_dst}, p0: #{p0}, p1: #{p1}, error: #{ex.message}"
        )
        raise KvCache::Error.new(error_msg)
      end
    end

    # Keeps only the specified sequence in the KV cache, removing all others
    #
    # Parameters:
    # - seq_id: The sequence ID to keep
    #
    # Returns:
    # - self for method chaining
    #
    # Raises:
    # - Llama::KvCache::Error if the operation fails
    def seq_keep(seq_id : Int32)
      begin
        LibLlama.llama_kv_self_seq_keep(ctx_ptr, seq_id)
        self
      rescue ex
        error_msg = Llama.format_error(
          "Failed to keep sequence in KV cache",
          -7, # KV cache error
          "seq_id: #{seq_id}, error: #{ex.message}"
        )
        raise KvCache::Error.new(error_msg)
      end
    end

    # Adds a relative position delta to tokens in a sequence
    #
    # Parameters:
    # - seq_id: The sequence ID to modify
    # - p0: Start position (p0 < 0 means start from 0)
    # - p1: End position (p1 < 0 means end at infinity)
    # - delta: The position delta to add
    #
    # Returns:
    # - self for method chaining
    #
    # Raises:
    # - Llama::KvCache::Error if the operation fails
    def seq_add(seq_id : Int32, p0 : Int32, p1 : Int32, delta : Int32)
      begin
        LibLlama.llama_kv_self_seq_add(ctx_ptr, seq_id, p0, p1, delta)
        self
      rescue ex
        error_msg = Llama.format_error(
          "Failed to add position delta to sequence in KV cache",
          -7, # KV cache error
          "seq_id: #{seq_id}, p0: #{p0}, p1: #{p1}, delta: #{delta}, error: #{ex.message}"
        )
        raise KvCache::Error.new(error_msg)
      end
    end

    # Divides the positions of tokens in a sequence by a factor
    #
    # Parameters:
    # - seq_id: The sequence ID to modify
    # - p0: Start position (p0 < 0 means start from 0)
    # - p1: End position (p1 < 0 means end at infinity)
    # - d: The divisor (must be > 1)
    #
    # Returns:
    # - self for method chaining
    #
    # Raises:
    # - ArgumentError if the divisor is not greater than 1
    # - Llama::KvCache::Error if the operation fails
    def seq_div(seq_id : Int32, p0 : Int32, p1 : Int32, d : Int32)
      if d <= 1
        raise ArgumentError.new("Divisor must be greater than 1")
      end

      begin
        LibLlama.llama_kv_self_seq_div(ctx_ptr, seq_id, p0, p1, d)
        self
      rescue ex
        error_msg = Llama.format_error(
          "Failed to divide positions in sequence in KV cache",
          -7, # KV cache error
          "seq_id: #{seq_id}, p0: #{p0}, p1: #{p1}, d: #{d}, error: #{ex.message}"
        )
        raise KvCache::Error.new(error_msg)
      end
    end

    # Returns the maximum position in a sequence
    #
    # Parameters:
    # - seq_id: The sequence ID to query
    #
    # Returns:
    # - The maximum position in the sequence
    #
    # Raises:
    # - Llama::KvCache::Error if the operation fails
    def seq_pos_max(seq_id : Int32) : Int32
      begin
        result = LibLlama.llama_kv_self_seq_pos_max(ctx_ptr, seq_id)

        if result < 0
          error_msg = Llama.format_error(
            "Failed to get maximum position in sequence",
            -7, # KV cache error
            "seq_id: #{seq_id}, result: #{result}"
          )
          raise KvCache::Error.new(error_msg)
        end

        result
      rescue ex : KvCache::Error
        raise ex
      rescue ex
        error_msg = Llama.format_error(
          "Failed to get maximum position in sequence",
          -7, # KV cache error
          "seq_id: #{seq_id}, error: #{ex.message}"
        )
        raise KvCache::Error.new(error_msg)
      end
    end

    # Defragments the KV cache
    # This will be applied lazily on next decode or explicitly with update
    #
    # Returns:
    # - self for method chaining
    #
    # Raises:
    # - Llama::KvCache::Error if the operation fails
    def defrag
      begin
        LibLlama.llama_kv_self_defrag(ctx_ptr)
        self
      rescue ex
        error_msg = Llama.format_error(
          "Failed to defragment KV cache",
          -7, # KV cache error
          ex.message
        )
        raise KvCache::Error.new(error_msg)
      end
    end

    # Checks if the context supports KV cache shifting
    #
    # Returns:
    # - true if the context supports KV cache shifting, false otherwise
    #
    # Raises:
    # - Llama::KvCache::Error if the operation fails
    def can_shift? : Bool
      begin
        LibLlama.llama_kv_self_can_shift(ctx_ptr)
      rescue ex
        error_msg = Llama.format_error(
          "Failed to check if context supports KV cache shifting",
          -7, # KV cache error
          ex.message
        )
        raise KvCache::Error.new(error_msg)
      end
    end

    # Applies pending KV cache updates
    # This includes K-shifts, defragmentation, etc.
    #
    # Returns:
    # - self for method chaining
    #
    # Raises:
    # - Llama::KvCache::Error if the operation fails
    def update
      begin
        LibLlama.llama_kv_self_update(ctx_ptr)
        self
      rescue ex
        error_msg = Llama.format_error(
          "Failed to update KV cache",
          -7, # KV cache error
          ex.message
        )
        raise KvCache::Error.new(error_msg)
      end
    end

    # Returns the raw pointer to the underlying llama_kv_cache structure
    def to_unsafe
      @handle
    end

    # Frees the resources associated with this KV cache
    def finalize
      # We don't own the handle, so we don't free it
      # Just nullify our references
      @ctx_ptr = Pointer(LibLlama::LlamaContext).null
    end

    @handle : LibLlama::LlamaKvCache*
    @ctx_ptr : LibLlama::LlamaContext*

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
