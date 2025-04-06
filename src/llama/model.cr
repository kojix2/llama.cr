module Llama
  # Wrapper for the llama_model structure
  class Model
    # Creates a new Model instance by loading a model from a file
    #
    # Parameters:
    # - path: Path to the model file (.gguf format)
    #
    # Raises:
    # - Llama::ModelError if the model cannot be loaded
    def initialize(path : String)
      params = LibLlama.llama_model_default_params
      @handle = LibLlama.llama_model_load_from_file(path, params)

      if @handle.null?
        error_msg = Llama.format_error(
          "Failed to load model",
          -5, # Model loading error
          "path: #{path}"
        )
        raise ModelError.new(error_msg)
      end
    end

    # Gets the default chat template for this model
    #
    # Parameters:
    # - name: Optional template name (nil for default)
    #
    # Returns:
    # - The chat template string, or nil if not available
    def chat_template(name : String? = nil) : String?
      ptr = LibLlama.llama_model_chat_template(@handle, name.nil? ? nil : name.to_unsafe)
      if ptr.null?
        # No error, just no template available
        return nil
      end
      String.new(ptr)
    end

    # Returns the vocabulary associated with this model
    def vocab : Vocab
      vocab_ptr = LibLlama.llama_model_get_vocab(@handle)
      if vocab_ptr.null?
        error_msg = Llama.format_error(
          "Failed to get vocabulary",
          -5, # Model error
          "model may be corrupted"
        )
        raise ModelError.new(error_msg)
      end
      Vocab.new(vocab_ptr)
    end

    # Returns the number of parameters in the model
    def n_params : UInt64
      LibLlama.llama_model_n_params(@handle)
    end

    # Returns the number of embedding dimensions in the model
    def n_embd : Int32
      LibLlama.llama_model_n_embd(@handle)
    end

    # Returns the number of layers in the model
    def n_layer : Int32
      LibLlama.llama_model_n_layer(@handle)
    end

    # Returns the number of attention heads in the model
    def n_head : Int32
      LibLlama.llama_model_n_head(@handle)
    end

    # Returns whether the model contains an encoder
    def has_encoder? : Bool
      LibLlama.llama_model_has_encoder(@handle)
    end

    # Returns whether the model contains a decoder
    def has_decoder? : Bool
      LibLlama.llama_model_has_decoder(@handle)
    end

    # Returns whether the model is recurrent (like Mamba, RWKV, etc.)
    def recurrent? : Bool
      LibLlama.llama_model_is_recurrent(@handle)
    end

    # Returns the model's RoPE frequency scaling factor
    def rope_freq_scale_train : Float32
      LibLlama.llama_model_rope_freq_scale_train(@handle)
    end

    # Returns the token that must be provided to the decoder to start generating output
    # For encoder-decoder models, returns the decoder start token
    # For other models, returns -1
    def decoder_start_token : Int32
      LibLlama.llama_model_decoder_start_token(@handle)
    end

    # Creates a new Context for this model
    #
    # Parameters:
    # - params: Optional context parameters (nil for default)
    #
    # Returns:
    # - A new Context instance
    #
    # Raises:
    # - Llama::ContextError if the context cannot be created
    def context(params = nil) : Context
      params ||= Llama::Context.default_params
      Context.new(self, params)
    end

    # Returns the raw pointer to the underlying llama_model structure
    def to_unsafe
      @handle
    end

    # Explicitly clean up resources
    # This can be called manually to release resources before garbage collection
    def cleanup
      if @handle && !@handle.null?
        LibLlama.llama_model_free(@handle)
      end
    end

    # Frees the resources associated with this model
    def finalize
      cleanup
    end

    # ===== MODEL METADATA METHODS =====

    # Gets a metadata value as a string by key name
    #
    # Parameters:
    # - key: The metadata key to look up
    #
    # Returns:
    # - The metadata value as a string, or nil if not found
    def meta_val_str(key : String) : String?
      buf_size = 1024
      buf = Pointer(LibC::Char).malloc(buf_size)
      result = LibLlama.llama_model_meta_val_str(@handle, key, buf, buf_size)

      if result < 0
        # Not an error, just no metadata found
        nil
      else
        String.new(buf, result)
      end
    end

    # Gets the number of metadata key/value pairs
    #
    # Returns:
    # - The number of metadata entries
    def meta_count : Int32
      LibLlama.llama_model_meta_count(@handle)
    end

    # Gets a metadata key name by index
    #
    # Parameters:
    # - i: The index of the metadata entry
    #
    # Returns:
    # - The key name, or nil if the index is out of bounds
    def meta_key_by_index(i : Int32) : String?
      buf_size = 1024
      buf = Pointer(LibC::Char).malloc(buf_size)
      result = LibLlama.llama_model_meta_key_by_index(@handle, i, buf, buf_size)

      if result < 0
        # Not an error, just index out of bounds
        nil
      else
        String.new(buf, result)
      end
    end

    # Gets a metadata value as a string by index
    #
    # Parameters:
    # - i: The index of the metadata entry
    #
    # Returns:
    # - The value as a string, or nil if the index is out of bounds
    def meta_val_str_by_index(i : Int32) : String?
      buf_size = 1024
      buf = Pointer(LibC::Char).malloc(buf_size)
      result = LibLlama.llama_model_meta_val_str_by_index(@handle, i, buf, buf_size)

      if result < 0
        # Not an error, just index out of bounds
        nil
      else
        String.new(buf, result)
      end
    end

    # Gets a string describing the model type
    #
    # Returns:
    # - A description of the model
    def description : String
      buf_size = 1024
      buf = Pointer(LibC::Char).malloc(buf_size)
      result = LibLlama.llama_model_desc(@handle, buf, buf_size)

      if result < 0
        # Error getting description, return a default
        "Unknown model"
      else
        String.new(buf, result)
      end
    end

    # Gets all metadata as a hash
    #
    # Returns:
    # - A hash mapping metadata keys to values
    def metadata : Hash(String, String)
      result = {} of String => String
      count = meta_count

      count.times do |i|
        key = meta_key_by_index(i)
        val = meta_val_str_by_index(i)

        if key && val
          result[key] = val
        end
      end

      result
    end

    @handle : LibLlama::LlamaModel*
  end
end
