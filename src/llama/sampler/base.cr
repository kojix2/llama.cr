module Llama
  module Sampler
    # Base class for all sampling methods.
    # Sampling is the process of selecting the next token during text generation.
    # This is an abstract class.
    abstract class Base
      # Creates a new Sampler instance from a raw pointer.
      #
      # Note: This constructor is intended for internal use.
      def initialize(@handle : LibLlama::LlamaSampler*)
        @owned_by_chain = false
      end

      # Returns the raw pointer to the underlying llama_sampler structure.
      def to_unsafe
        @handle
      end

      # :nodoc:
      # Mark this sampler as owned by a SamplerChain (prevents double free)
      # Do not call this method directly.
      protected def set_owned_by_chain(v : Bool = true)
        @owned_by_chain = v
      end

      # :nodoc:
      # Sets the handle to a new pointer.
      # This is used when a sampler is removed from a chain and needs to be
      protected def set_handle(ptr : LibLlama::LlamaSampler*)
        @handle = ptr
      end

      # :nodoc:
      # Frees the resources associated with this sampler.
      def finalize
        if !@owned_by_chain && @handle && !@handle.null?
          LibLlama.llama_sampler_free(@handle)
        end
      end

      @handle : LibLlama::LlamaSampler*
      @owned_by_chain : Bool

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
end
