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

      # Mark this sampler as owned by a SamplerChain (prevents double free)
      # Do not call this method directly.
      def mark_owned_by_chain
        @owned_by_chain = true
      end

      # Frees the resources associated with this sampler.
      def finalize
        if !@owned_by_chain && @handle && !@handle.null?
          LibLlama.llama_sampler_free(@handle)
        end
      end

      @handle : LibLlama::LlamaSampler*
      @owned_by_chain : Bool
    end
  end
end
