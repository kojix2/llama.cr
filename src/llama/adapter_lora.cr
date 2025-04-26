require "./adapter_lora/error"

module Llama
  # Wrapper for the llama_adapter_lora structure
  #
  # This class represents a LoRA (Low-Rank Adaptation) adapter that can be
  # applied to a model to modify its behavior without changing the original weights.
  class AdapterLora
    # Creates a new LoRA adapter from a file
    #
    # Parameters:
    # - model: The Model to load the adapter for
    # - path: Path to the LoRA adapter file
    #
    # Raises:
    # - Llama::AdapterLora::Error if the adapter cannot be loaded
    def initialize(model : Model, path : String)
      # Ensure llama backend is initialized
      Llama.init

      @handle = LibLlama.llama_adapter_lora_init(model.to_unsafe, path)

      if @handle.null?
        error_msg = "Failed to load LoRA adapter from '#{path}'"
        raise AdapterLora::Error.new(error_msg)
      end
    end

    # Explicitly clean up resources
    # This can be called manually to release resources before garbage collection
    private def cleanup
      if @handle && !@handle.null?
        LibLlama.llama_adapter_lora_free(@handle)
        @handle = Pointer(Void).null
      end
    end

    # Returns the raw pointer to the underlying llama_adapter_lora structure
    def to_unsafe
      @handle
    end

    # Frees the resources associated with this adapter
    def finalize
      cleanup
    end

    # :nodoc:
    def clone
      raise NotImplementedError.new("clone is not supported for #{self.class}")
    end

    # :nodoc:
    def dup
      raise NotImplementedError.new("dup is not supported for #{self.class}")
    end

    @handle : LibLlama::LlamaAdapterLora*
  end
end
