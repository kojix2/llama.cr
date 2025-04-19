module Llama
  # Wrapper for state management functions in llama.cpp
  # Provides methods for saving and loading model state
  class State
    # Creates a new State instance
    #
    # Parameters:
    # - ctx: The Context to manage state for
    #
    # To avoid circular references, we store the context pointer rather than the context object
    #
    # Raises:
    # - Llama::State::Error if the context pointer is null
    def initialize(ctx : Context)
      @ctx_ptr = ctx.to_unsafe

      if @ctx_ptr.null?
        error_msg = Llama.format_error(
          "Failed to initialize state",
          -8, # State management error
          "context pointer is null"
        )
        raise State::Error.new(error_msg)
      end
    end

    # Get the context pointer for internal use
    private def ctx_ptr : LibLlama::LlamaContext*
      if @ctx_ptr.null?
        error_msg = Llama.format_error(
          "Invalid context pointer",
          -8, # State management error
          "context pointer is null"
        )
        raise State::Error.new(error_msg)
      end

      @ctx_ptr
    end

    # Returns the size in bytes needed to store the current state
    #
    # Returns:
    # - The size in bytes
    #
    # Raises:
    # - Llama::State::Error if the operation fails
    def size : LibC::SizeT
      begin
        result = LibLlama.llama_state_get_size(ctx_ptr)

        if result == 0
          error_msg = Llama.format_error(
            "Failed to get state size",
            -8, # State management error
            "returned size is 0"
          )
          raise State::Error.new(error_msg)
        end

        result
      rescue ex : State::Error
        raise ex
      rescue ex
        error_msg = Llama.format_error(
          "Failed to get state size",
          -8, # State management error
          ex.message
        )
        raise State::Error.new(error_msg)
      end
    end

    # Gets the current state data
    #
    # Returns:
    # - A Bytes object containing the state data
    #
    # Raises:
    # - Llama::State::Error if the operation fails
    def get_data : Bytes
      begin
        # Get the size needed
        state_size = size

        # Allocate a buffer
        buffer = Bytes.new(state_size)

        # Get the state data
        bytes_copied = LibLlama.llama_state_get_data(ctx_ptr, buffer.to_unsafe, state_size)

        if bytes_copied == 0
          error_msg = Llama.format_error(
            "Failed to get state data",
            -8, # State management error
            "no bytes were copied"
          )
          raise State::Error.new(error_msg)
        end

        # Return the buffer (potentially truncated if bytes_copied < state_size)
        buffer[0, bytes_copied]
      rescue ex : State::Error
        raise ex
      rescue ex
        error_msg = Llama.format_error(
          "Failed to get state data",
          -8, # State management error
          ex.message
        )
        raise State::Error.new(error_msg)
      end
    end

    # Sets the state from data
    #
    # Parameters:
    # - data: The state data to set
    #
    # Returns:
    # - The number of bytes read
    #
    # Raises:
    # - ArgumentError if data is empty
    # - Llama::State::Error if the operation fails
    def set_data(data : Bytes) : LibC::SizeT
      if data.empty?
        raise ArgumentError.new("State data cannot be empty")
      end

      begin
        result = LibLlama.llama_state_set_data(ctx_ptr, data.to_unsafe, data.size)

        if result == 0
          error_msg = Llama.format_error(
            "Failed to set state data",
            -8, # State management error
            "no bytes were read"
          )
          raise State::Error.new(error_msg)
        end

        result
      rescue ex : State::Error | ArgumentError
        raise ex
      rescue ex
        error_msg = Llama.format_error(
          "Failed to set state data",
          -8, # State management error
          ex.message
        )
        raise State::Error.new(error_msg)
      end
    end

    # Loads state from a file
    #
    # Parameters:
    # - path: Path to the session file
    # - max_tokens: Maximum number of tokens to load (default: 1024)
    #
    # Returns:
    # - An array of tokens loaded from the file
    #
    # Raises:
    # - ArgumentError if path is empty or max_tokens is not positive
    # - Llama::State::Error if the file cannot be loaded
    def load_file(path : String, max_tokens : Int32 = 1024) : Array(Int32)
      if path.empty?
        raise ArgumentError.new("Path cannot be empty")
      end

      if max_tokens <= 0
        raise ArgumentError.new("max_tokens must be positive")
      end

      begin
        # Allocate a buffer for tokens
        tokens = Pointer(LibLlama::LlamaToken).malloc(max_tokens)
        token_count = Pointer(LibC::SizeT).malloc(1)
        token_count.value = 0

        # Load the file
        success = LibLlama.llama_state_load_file(
          ctx_ptr,
          path,
          tokens,
          max_tokens,
          token_count
        )

        unless success
          error_msg = Llama.format_error(
            "Failed to load state from file",
            -8, # State management error
            "path: #{path}"
          )
          raise State::Error.new(error_msg)
        end

        # Convert to Crystal array
        result = Array(Int32).new(token_count.value)
        token_count.value.times do |i|
          result << tokens[i]
        end

        result
      rescue ex : State::Error | ArgumentError
        raise ex
      rescue ex
        error_msg = Llama.format_error(
          "Failed to load state from file",
          -8, # State management error
          "path: #{path}, error: #{ex.message}"
        )
        raise State::Error.new(error_msg)
      end
    end

    # Saves state to a file
    #
    # Parameters:
    # - path: Path to save the session file
    # - tokens: Array of tokens to save with the state
    #
    # Returns:
    # - true if successful, false otherwise
    #
    # Raises:
    # - ArgumentError if path is empty
    # - Llama::State::Error if the operation fails
    def save_file(path : String, tokens : Array(Int32)) : Bool
      if path.empty?
        raise ArgumentError.new("Path cannot be empty")
      end

      begin
        # Convert tokens to C array
        tokens_ptr = tokens.to_unsafe

        # Save the file
        result = LibLlama.llama_state_save_file(
          ctx_ptr,
          path,
          tokens_ptr,
          tokens.size
        )

        unless result
          error_msg = Llama.format_error(
            "Failed to save state to file",
            -8, # State management error
            "path: #{path}"
          )
          raise State::Error.new(error_msg)
        end

        result
      rescue ex : State::Error | ArgumentError
        raise ex
      rescue ex
        error_msg = Llama.format_error(
          "Failed to save state to file",
          -8, # State management error
          "path: #{path}, error: #{ex.message}"
        )
        raise State::Error.new(error_msg)
      end
    end

    # Gets the size needed to store a specific sequence's state
    #
    # Parameters:
    # - seq_id: The sequence ID
    #
    # Returns:
    # - The size in bytes
    #
    # Raises:
    # - Llama::State::Error if the operation fails
    def seq_size(seq_id : Int32) : LibC::SizeT
      begin
        result = LibLlama.llama_state_seq_get_size(ctx_ptr, seq_id)

        if result == 0
          error_msg = Llama.format_error(
            "Failed to get sequence state size",
            -8, # State management error
            "seq_id: #{seq_id}, returned size is 0"
          )
          raise State::Error.new(error_msg)
        end

        result
      rescue ex : State::Error
        raise ex
      rescue ex
        error_msg = Llama.format_error(
          "Failed to get sequence state size",
          -8, # State management error
          "seq_id: #{seq_id}, error: #{ex.message}"
        )
        raise State::Error.new(error_msg)
      end
    end

    # Gets the state data for a specific sequence
    #
    # Parameters:
    # - seq_id: The sequence ID
    #
    # Returns:
    # - A Bytes object containing the sequence state data
    #
    # Raises:
    # - Llama::State::Error if the operation fails
    def seq_get_data(seq_id : Int32) : Bytes
      begin
        # Get the size needed
        state_size = seq_size(seq_id)

        # Allocate a buffer
        buffer = Bytes.new(state_size)

        # Get the state data
        bytes_copied = LibLlama.llama_state_seq_get_data(
          ctx_ptr,
          buffer.to_unsafe,
          state_size,
          seq_id
        )

        if bytes_copied == 0
          error_msg = Llama.format_error(
            "Failed to get sequence state data",
            -8, # State management error
            "seq_id: #{seq_id}, no bytes were copied"
          )
          raise State::Error.new(error_msg)
        end

        # Return the buffer (potentially truncated if bytes_copied < state_size)
        buffer[0, bytes_copied]
      rescue ex : State::Error
        raise ex
      rescue ex
        error_msg = Llama.format_error(
          "Failed to get sequence state data",
          -8, # State management error
          "seq_id: #{seq_id}, error: #{ex.message}"
        )
        raise State::Error.new(error_msg)
      end
    end

    # Sets the state for a specific sequence from data
    #
    # Parameters:
    # - data: The state data to set
    # - dest_seq_id: The destination sequence ID
    #
    # Returns:
    # - The number of bytes read, or 0 if failed
    #
    # Raises:
    # - ArgumentError if data is empty
    # - Llama::State::Error if the operation fails
    def seq_set_data(data : Bytes, dest_seq_id : Int32) : LibC::SizeT
      if data.empty?
        raise ArgumentError.new("State data cannot be empty")
      end

      begin
        result = LibLlama.llama_state_seq_set_data(
          ctx_ptr,
          data.to_unsafe,
          data.size,
          dest_seq_id
        )

        if result == 0
          error_msg = Llama.format_error(
            "Failed to set sequence state data",
            -8, # State management error
            "dest_seq_id: #{dest_seq_id}, no bytes were read"
          )
          raise State::Error.new(error_msg)
        end

        result
      rescue ex : State::Error | ArgumentError
        raise ex
      rescue ex
        error_msg = Llama.format_error(
          "Failed to set sequence state data",
          -8, # State management error
          "dest_seq_id: #{dest_seq_id}, error: #{ex.message}"
        )
        raise State::Error.new(error_msg)
      end
    end

    # Saves a sequence's state to a file
    #
    # Parameters:
    # - path: Path to save the sequence file
    # - seq_id: The sequence ID to save
    # - tokens: Array of tokens to save with the state
    #
    # Returns:
    # - The number of bytes written, or 0 if failed
    #
    # Raises:
    # - ArgumentError if path is empty
    # - Llama::State::Error if the operation fails
    def seq_save_file(path : String, seq_id : Int32, tokens : Array(Int32)) : LibC::SizeT
      if path.empty?
        raise ArgumentError.new("Path cannot be empty")
      end

      begin
        # Convert tokens to C array
        tokens_ptr = tokens.to_unsafe

        # Save the file
        result = LibLlama.llama_state_seq_save_file(
          ctx_ptr,
          path,
          seq_id,
          tokens_ptr,
          tokens.size
        )

        if result == 0
          error_msg = Llama.format_error(
            "Failed to save sequence state to file",
            -8, # State management error
            "path: #{path}, seq_id: #{seq_id}"
          )
          raise State::Error.new(error_msg)
        end

        result
      rescue ex : State::Error | ArgumentError
        raise ex
      rescue ex
        error_msg = Llama.format_error(
          "Failed to save sequence state to file",
          -8, # State management error
          "path: #{path}, seq_id: #{seq_id}, error: #{ex.message}"
        )
        raise State::Error.new(error_msg)
      end
    end

    # Loads a sequence's state from a file
    #
    # Parameters:
    # - path: Path to the sequence file
    # - dest_seq_id: The destination sequence ID
    # - max_tokens: Maximum number of tokens to load (default: 1024)
    #
    # Returns:
    # - An array of tokens loaded from the file
    #
    # Raises:
    # - ArgumentError if path is empty or max_tokens is not positive
    # - Llama::State::Error if the file cannot be loaded
    def seq_load_file(path : String, dest_seq_id : Int32, max_tokens : Int32 = 1024) : Array(Int32)
      if path.empty?
        raise ArgumentError.new("Path cannot be empty")
      end

      if max_tokens <= 0
        raise ArgumentError.new("max_tokens must be positive")
      end

      begin
        # Allocate a buffer for tokens
        tokens = Pointer(LibLlama::LlamaToken).malloc(max_tokens)
        token_count = Pointer(LibC::SizeT).malloc(1)
        token_count.value = 0

        # Load the file
        bytes_read = LibLlama.llama_state_seq_load_file(
          ctx_ptr,
          path,
          dest_seq_id,
          tokens,
          max_tokens,
          token_count
        )

        if bytes_read == 0
          error_msg = Llama.format_error(
            "Failed to load sequence state from file",
            -8, # State management error
            "path: #{path}, dest_seq_id: #{dest_seq_id}"
          )
          raise State::Error.new(error_msg)
        end

        # Convert to Crystal array
        result = Array(Int32).new(token_count.value)
        token_count.value.times do |i|
          result << tokens[i]
        end

        result
      rescue ex : State::Error | ArgumentError
        raise ex
      rescue ex
        error_msg = Llama.format_error(
          "Failed to load sequence state from file",
          -8, # State management error
          "path: #{path}, dest_seq_id: #{dest_seq_id}, error: #{ex.message}"
        )
        raise State::Error.new(error_msg)
      end
    end

    # Frees the resources associated with this state
    def finalize
      # Just nullify our references
      @ctx_ptr = Pointer(LibLlama::LlamaContext).null
    end

    @ctx_ptr : LibLlama::LlamaContext*
  end
end
