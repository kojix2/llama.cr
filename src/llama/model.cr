module Llama
  # Represents an error that occurred in the Llama library
  class Error < Exception; end

  # Wrapper for the llama_model structure
  class Model
    # Creates a new Model instance by loading a model from a file
    #
    # Parameters:
    # - path: Path to the model file (.gguf format)
    #
    # Raises:
    # - Llama::Error if the model cannot be loaded
    def initialize(path : String)
      params = LibLlama.llama_model_default_params
      @handle = LibLlama.llama_model_load_from_file(path, params)
      raise Error.new("Failed to load model from #{path}") if @handle.null?
    end

    # Returns the vocabulary associated with this model
    def vocab : Vocab
      vocab_ptr = LibLlama.llama_model_get_vocab(@handle)
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

    # Creates a new Context for this model
    def context(params = nil) : Context
      params ||= LibLlama.llama_context_default_params
      Context.new(self, params)
    end

    # Returns the raw pointer to the underlying llama_model structure
    def to_unsafe
      @handle
    end

    # Frees the resources associated with this model
    def finalize
      LibLlama.llama_model_free(@handle) unless @handle.null?
    end

    @handle : LibLlama::LlamaModel*
  end
end
