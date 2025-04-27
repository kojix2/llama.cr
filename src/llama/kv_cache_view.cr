require "./kv_cache_view/error"

module Llama
  # Wrapper for the llama_kv_cache_view_cell structure
  # Represents a single cell in the KV cache
  class KvCacheViewCell
    # Creates a new KvCacheViewCell instance
    #
    # Parameters:
    # - cell: The raw llama_kv_cache_view_cell structure
    def initialize(@cell : LibLlama::LlamaKvCacheViewCell)
    end

    # Returns the position for this cell
    #
    # Returns:
    # - The position for this cell
    def pos : Int32
      @cell.pos
    end

    # Returns a string representation of the cell
    #
    # Returns:
    # - A string representation of the cell
    def to_s(io : IO) : Nil
      io << "KvCacheViewCell(pos: #{pos})"
    end

    # Returns a detailed string representation of the cell
    #
    # Returns:
    # - A detailed string representation of the cell
    def inspect(io : IO) : Nil
      to_s(io)
    end

    @cell : LibLlama::LlamaKvCacheViewCell
  end

  # Wrapper for the llama_kv_cache_view structure
  # Provides methods for visualizing and debugging the KV cache state
  class KvCacheView
    include Enumerable(KvCacheViewCell)

    # Creates a new KvCacheView instance
    #
    # Parameters:
    # - ctx: The context to create the view for
    # - n_seq_max: Maximum number of sequences per cell to track (default: 4)
    #
    # Raises:
    # - Llama::KvCacheView::Error if the view cannot be created
    def initialize(ctx : Context, n_seq_max : Int32 = 4)
      if ctx.to_unsafe.null?
        raise Error.new("Invalid context: null pointer")
      end

      @view = LibLlama.llama_kv_cache_view_init(ctx.to_unsafe, n_seq_max)
      @ctx_ptr = ctx.to_unsafe
      @freed = false

      if @view.n_cells < 0
        @freed = true
        raise Error.new("Failed to initialize KV cache view")
      end

      # Update the view to get the current state of the KV cache
      update
    end

    # Creates a new KvCacheView instance with a block
    #
    # Parameters:
    # - ctx: The context to create the view for
    # - n_seq_max: Maximum number of sequences per cell to track (default: 4)
    # - block: The block to execute with the view
    #
    # The view will be automatically freed after the block execution
    #
    # Raises:
    # - Llama::KvCacheView::Error if the view cannot be created
    def initialize(ctx : Context, n_seq_max : Int32 = 4, &block)
      initialize(ctx, n_seq_max)
      begin
        yield self
      ensure
        free unless freed?
      end
    end

    # Returns whether the view has been freed
    #
    # Returns:
    # - true if the view has been freed, false otherwise
    def freed? : Bool
      @freed
    end

    # Updates the view with the current state of the KV cache
    #
    # Returns:
    # - self for method chaining
    #
    # Raises:
    # - Llama::KvCacheView::Error if the update fails
    def update : self
      check_freed

      begin
        LibLlama.llama_kv_cache_view_update(@ctx_ptr, pointerof(@view))
        self
      rescue ex
        error_msg = Llama.format_error(
          "Failed to update KV cache view",
          -7, # KV cache error
          ex.message
        )
        raise Error.new(error_msg)
      end
    end

    # Frees the resources associated with this view
    #
    # Returns:
    # - self for method chaining
    #
    # Raises:
    # - Llama::KvCacheView::Error if the free operation fails
    def free : self
      return self if @freed

      begin
        LibLlama.llama_kv_cache_view_free(pointerof(@view))
        @freed = true
        self
      rescue ex
        error_msg = Llama.format_error(
          "Failed to free KV cache view",
          -7, # KV cache error
          ex.message
        )
        raise Error.new(error_msg)
      end
    end

    # Returns the number of cells in the KV cache
    #
    # Returns:
    # - The number of cells
    #
    # Raises:
    # - Llama::KvCacheView::Error if the view has been freed
    def n_cells : Int32
      check_freed
      @view.n_cells
    end

    # Returns the maximum number of sequences per cell
    #
    # Returns:
    # - The maximum number of sequences per cell
    #
    # Raises:
    # - Llama::KvCacheView::Error if the view has been freed
    def n_seq_max : Int32
      check_freed
      @view.n_seq_max
    end

    # Returns the total number of tokens in the KV cache
    #
    # Returns:
    # - The number of tokens
    #
    # Raises:
    # - Llama::KvCacheView::Error if the view has been freed
    def token_count : Int32
      check_freed
      @view.token_count
    end

    # Returns the number of used cells in the KV cache
    #
    # Returns:
    # - The number of used cells
    #
    # Raises:
    # - Llama::KvCacheView::Error if the view has been freed
    def used_cells : Int32
      check_freed
      @view.used_cells
    end

    # Returns the maximum number of contiguous empty slots
    #
    # Returns:
    # - The maximum number of contiguous empty slots
    #
    # Raises:
    # - Llama::KvCacheView::Error if the view has been freed
    def max_contiguous : Int32
      check_freed
      @view.max_contiguous
    end

    # Returns the index to the start of the max_contiguous slot range
    #
    # Returns:
    # - The index to the start of the max_contiguous slot range
    #
    # Raises:
    # - Llama::KvCacheView::Error if the view has been freed
    def max_contiguous_idx : Int32
      check_freed
      @view.max_contiguous_idx
    end

    # Returns whether the KV cache is empty
    #
    # Returns:
    # - true if the KV cache is empty, false otherwise
    #
    # Raises:
    # - Llama::KvCacheView::Error if the view has been freed
    def empty? : Bool
      check_freed
      token_count == 0
    end

    # Returns whether the KV cache is full
    #
    # Returns:
    # - true if the KV cache is full, false otherwise
    #
    # Raises:
    # - Llama::KvCacheView::Error if the view has been freed
    def full? : Bool
      check_freed
      used_cells == n_cells
    end

    # Alias for token_count
    def size : Int32
      token_count
    end

    # Alias for n_cells
    def capacity : Int32
      n_cells
    end

    # Implements Enumerable interface
    # Yields each cell in the KV cache
    #
    # Raises:
    # - Llama::KvCacheView::Error if the view has been freed
    def each(&)
      check_freed

      n_cells.times do |i|
        yield KvCacheViewCell.new(@view.cells[i])
      end
    end

    # Returns the cell at the specified index
    #
    # Parameters:
    # - index: The index of the cell to get
    #
    # Returns:
    # - The cell at the specified index
    #
    # Raises:
    # - IndexError if the index is out of bounds
    # - Llama::KvCacheView::Error if the view has been freed
    def [](index : Int32) : KvCacheViewCell
      check_freed

      if index < 0 || index >= n_cells
        raise IndexError.new("Index out of bounds: #{index} (valid range: 0..#{n_cells - 1})")
      end

      KvCacheViewCell.new(@view.cells[index])
    end

    # Returns the sequences for the cell at the specified index
    #
    # Parameters:
    # - index: The index of the cell to get sequences for
    #
    # Returns:
    # - An array of sequence IDs for the cell
    #
    # Raises:
    # - IndexError if the index is out of bounds
    # - Llama::KvCacheView::Error if the view has been freed
    def sequences(index : Int32) : Array(Int32)
      check_freed

      if index < 0 || index >= n_cells
        raise IndexError.new("Index out of bounds: #{index} (valid range: 0..#{n_cells - 1})")
      end

      # Calculate the offset in the cells_sequences array
      offset = index * n_seq_max

      # Collect all non-negative sequence IDs
      result = [] of Int32
      n_seq_max.times do |i|
        seq_id = @view.cells_sequences[offset + i]
        break if seq_id < 0
        result << seq_id
      end

      result
    end

    # Returns a string representation of the KV cache view
    #
    # Returns:
    # - A string representation of the KV cache view
    def to_s(io : IO) : Nil
      if @freed
        io << "KvCacheView(freed)"
      else
        io << "KvCacheView(cells: #{n_cells}, used: #{used_cells}, tokens: #{token_count})"
      end
    end

    # Returns a detailed string representation of the KV cache view
    #
    # Returns:
    # - A detailed string representation of the KV cache view
    def inspect(io : IO) : Nil
      if @freed
        io << "KvCacheView(freed)"
      else
        io << "KvCacheView(\n"
        io << "  cells: #{n_cells},\n"
        io << "  used: #{used_cells},\n"
        io << "  tokens: #{token_count},\n"
        io << "  max_contiguous: #{max_contiguous},\n"
        io << "  max_contiguous_idx: #{max_contiguous_idx}\n"
        io << ")"
      end
    end

    # Internal check to ensure the view has not been freed
    private def check_freed
      if @freed
        raise Error.new("Attempt to use freed KV cache view")
      end

      if @ctx_ptr.null?
        @freed = true
        raise Error.new("Context has been freed")
      end
    end

    # Frees the resources associated with this view
    def finalize
      free unless @freed
    end

    @view : LibLlama::LlamaKvCacheView
    @ctx_ptr : LibLlama::LlamaContext*
    @freed : Bool

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
