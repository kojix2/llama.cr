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
      end

      # Returns the raw pointer to the underlying llama_sampler structure.
      def to_unsafe
        @handle
      end

      # Frees the resources associated with this sampler.
      def finalize
        if @handle && !@handle.null?
          LibLlama.llama_sampler_free(@handle)
        end
      end

      @handle : LibLlama::LlamaSampler*
    end
  end
end
