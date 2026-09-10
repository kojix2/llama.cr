require "./adapter_lora/error"

module Llama
  # Wrapper for the llama_adapter_lora structure
  #
  # This class represents a LoRA (Low-Rank Adaptation) adapter that can be
  # applied to a model to modify its behavior without changing the original weights.
  class AdapterLora
    include NativeResource

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

      # Keep the associated model alive while this adapter exists.
      # llama.cpp requires adapter lifetime to be within model lifetime.
      @model = model
      @attachments_mutex = Mutex.new
      @attachments = [] of WeakRef(Context)

      @handle = LibLlama.llama_adapter_lora_init(model.unsafe_handle!, path)

      if @handle.null?
        error_msg = "Failed to load LoRA adapter from '#{path}'"
        raise AdapterLora::Error.new(error_msg)
      end

      begin
        @model.register_child(self)
      rescue ex
        cleanup
        raise ex
      end
    end

    # Releases the underlying C resources
    #
    # Calling this method multiple times is safe; only the first call
    # releases the resources. The finalizer also calls this method, so
    # manual calls are only needed to release resources deterministically,
    # for example before process exit.
    def free : Nil
      close
    end

    # Releases the adapter unless it is still attached to a context.
    def close : Nil
      @attachments_mutex.synchronize do
        return if closed?
        raise BusyError.new(self.class.to_s) if @attachments.any?(&.value)
        cleanup
      end
    end

    # Returns whether the native adapter has been released.
    def closed? : Bool
      @handle.null?
    end

    # Returns a checked native handle for internal wrapper use.
    # :nodoc:
    def unsafe_handle! : LibLlama::LlamaAdapterLora*
      ensure_open!
      @handle
    end

    # :nodoc:
    def register_attachment(context : Context) : Nil
      @attachments_mutex.synchronize do
        ensure_open!
        unless @attachments.any? { |ref| ref.value.try(&.same?(context)) }
          @attachments << WeakRef.new(context)
        end
      end
    end

    # :nodoc:
    def unregister_attachment(context : Context) : Nil
      @attachments_mutex.synchronize do
        @attachments.reject! { |ref| value = ref.value; value.nil? || value.same?(context) }
      end
    end

    private def cleanup
      if @handle && !@handle.null?
        LibLlama.llama_adapter_lora_free(@handle)
        @handle = Pointer(LibLlama::LlamaAdapterLora).null
        @model.unregister_child(self)
      end
    end

    # Returns the raw pointer to the underlying llama_adapter_lora structure
    def to_unsafe
      @handle
    end

    # Returns the model this adapter was loaded for.
    getter model : Model

    # Frees the resources associated with this adapter
    def finalize
      close
    rescue
      # Finalizers are a best-effort fallback and must never raise.
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
    @model : Model
    @attachments_mutex : Mutex
    @attachments : Array(WeakRef(Context))
  end
end
