require "./memory/error"

module Llama
  # Modern memory management for llama.cpp contexts
  #
  # This class provides a unified interface to various memory types:
  # - Standard KV cache (llama_kv_cache_unified)
  # - SWA (Sliding Window Attention) cache (llama_kv_cache_unified_iswa)
  # - Recurrent layer memory (llama_memory_recurrent)
  # - Hybrid attention/recurrent models (llama_memory_hybrid)
  #
  # The Memory API replaces the deprecated KV cache API and provides
  # better support for modern model architectures.
  class Memory
    # Creates a new Memory instance from a context
    #
    # Parameters:
    # - ctx: The context to get memory from
    #
    # Raises:
    # - Memory::Error if memory handle cannot be obtained
    def initialize(ctx : Context)
      @handle = LibLlama.llama_get_memory(ctx.to_unsafe)
      if @handle.null?
        error_msg = Llama.format_error(
          "Failed to get memory handle",
          -7, # Memory error
          "context pointer may be invalid"
        )
        raise Memory::Error.new(error_msg)
      end
    end

    # Clear memory contents
    #
    # Parameters:
    # - data: If true, data buffers will also be cleared together with metadata (default: false)
    #
    # Returns:
    # - self for method chaining
    def clear(data : Bool = false) : self
      begin
        LibLlama.llama_memory_clear(@handle, data)
        self
      rescue ex
        error_msg = Llama.format_error(
          "Failed to clear memory",
          -7, # Memory error
          ex.message
        )
        raise Memory::Error.new(error_msg)
      end
    end

    # Remove tokens from sequence in specified position range
    #
    # Parameters:
    # - seq_id: Sequence ID (< 0 to match any sequence)
    # - p0: Start position (< 0 for [0, p1])
    # - p1: End position (< 0 for [p0, inf))
    #
    # Returns:
    # - true if successful, false if partial sequence cannot be removed
    #
    # Note: Removing a whole sequence never fails
    def seq_rm(seq_id : Int32, p0 : Int32, p1 : Int32) : Bool
      begin
        LibLlama.llama_memory_seq_rm(@handle, seq_id, p0, p1)
      rescue ex
        error_msg = Llama.format_error(
          "Failed to remove sequence from memory",
          -7, # Memory error
          "seq_id: #{seq_id}, p0: #{p0}, p1: #{p1}, error: #{ex.message}"
        )
        raise Memory::Error.new(error_msg)
      end
    end

    # Copy tokens from one sequence to another
    #
    # Parameters:
    # - seq_id_src: Source sequence ID
    # - seq_id_dst: Destination sequence ID
    # - p0: Start position (< 0 for [0, p1])
    # - p1: End position (< 0 for [p0, inf))
    #
    # Returns:
    # - self for method chaining
    def seq_cp(seq_id_src : Int32, seq_id_dst : Int32, p0 : Int32, p1 : Int32) : self
      begin
        LibLlama.llama_memory_seq_cp(@handle, seq_id_src, seq_id_dst, p0, p1)
        self
      rescue ex
        error_msg = Llama.format_error(
          "Failed to copy sequence in memory",
          -7, # Memory error
          "seq_id_src: #{seq_id_src}, seq_id_dst: #{seq_id_dst}, p0: #{p0}, p1: #{p1}, error: #{ex.message}"
        )
        raise Memory::Error.new(error_msg)
      end
    end

    # Keep only specified sequence, remove all others
    #
    # Parameters:
    # - seq_id: Sequence ID to keep
    #
    # Returns:
    # - self for method chaining
    def seq_keep(seq_id : Int32) : self
      begin
        LibLlama.llama_memory_seq_keep(@handle, seq_id)
        self
      rescue ex
        error_msg = Llama.format_error(
          "Failed to keep sequence in memory",
          -7, # Memory error
          "seq_id: #{seq_id}, error: #{ex.message}"
        )
        raise Memory::Error.new(error_msg)
      end
    end

    # Add relative position delta to tokens in sequence
    #
    # Parameters:
    # - seq_id: Sequence ID
    # - p0: Start position (< 0 for [0, p1])
    # - p1: End position (< 0 for [p0, inf))
    # - delta: Position delta to add
    #
    # Returns:
    # - self for method chaining
    def seq_add(seq_id : Int32, p0 : Int32, p1 : Int32, delta : Int32) : self
      begin
        LibLlama.llama_memory_seq_add(@handle, seq_id, p0, p1, delta)
        self
      rescue ex
        error_msg = Llama.format_error(
          "Failed to add position delta to sequence in memory",
          -7, # Memory error
          "seq_id: #{seq_id}, p0: #{p0}, p1: #{p1}, delta: #{delta}, error: #{ex.message}"
        )
        raise Memory::Error.new(error_msg)
      end
    end

    # Divide positions of tokens in sequence by factor
    #
    # Parameters:
    # - seq_id: Sequence ID
    # - p0: Start position (< 0 for [0, p1])
    # - p1: End position (< 0 for [p0, inf))
    # - d: Divisor (must be > 1)
    #
    # Returns:
    # - self for method chaining
    #
    # Raises:
    # - ArgumentError if divisor is <= 1
    def seq_div(seq_id : Int32, p0 : Int32, p1 : Int32, d : Int32) : self
      if d <= 1
        raise ArgumentError.new("Divisor must be greater than 1")
      end

      begin
        LibLlama.llama_memory_seq_div(@handle, seq_id, p0, p1, d)
        self
      rescue ex
        error_msg = Llama.format_error(
          "Failed to divide positions in sequence in memory",
          -7, # Memory error
          "seq_id: #{seq_id}, p0: #{p0}, p1: #{p1}, d: #{d}, error: #{ex.message}"
        )
        raise Memory::Error.new(error_msg)
      end
    end

    # Get minimum position in sequence
    #
    # This is typically non-zero only for SWA (Sliding Window Attention) caches.
    # All positions in the range [pos_min, pos_max] are guaranteed to be present.
    #
    # Parameters:
    # - seq_id: Sequence ID
    #
    # Returns:
    # - Minimum position, or -1 if sequence is empty
    def seq_pos_min(seq_id : Int32) : Int32
      begin
        LibLlama.llama_memory_seq_pos_min(@handle, seq_id)
      rescue ex
        error_msg = Llama.format_error(
          "Failed to get minimum position in sequence",
          -7, # Memory error
          "seq_id: #{seq_id}, error: #{ex.message}"
        )
        raise Memory::Error.new(error_msg)
      end
    end

    # Get maximum position in sequence
    #
    # All positions in the range [pos_min, pos_max] are guaranteed to be present.
    #
    # Parameters:
    # - seq_id: Sequence ID
    #
    # Returns:
    # - Maximum position, or -1 if sequence is empty
    def seq_pos_max(seq_id : Int32) : Int32
      begin
        LibLlama.llama_memory_seq_pos_max(@handle, seq_id)
      rescue ex
        error_msg = Llama.format_error(
          "Failed to get maximum position in sequence",
          -7, # Memory error
          "seq_id: #{seq_id}, error: #{ex.message}"
        )
        raise Memory::Error.new(error_msg)
      end
    end

    # Check if memory supports shifting
    #
    # Returns:
    # - true if shifting is supported, false otherwise
    def can_shift? : Bool
      begin
        LibLlama.llama_memory_can_shift(@handle)
      rescue ex
        error_msg = Llama.format_error(
          "Failed to check if memory supports shifting",
          -7, # Memory error
          ex.message
        )
        raise Memory::Error.new(error_msg)
      end
    end

    # Get raw pointer for internal use
    #
    # Returns:
    # - Raw memory handle pointer
    def to_unsafe
      @handle
    end

    @handle : LibLlama::LlamaMemoryT

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
