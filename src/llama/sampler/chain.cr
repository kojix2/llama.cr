module Llama
  module Sampler
    # Wrapper for a chain of samplers
    #
    # The SamplerChain class allows you to combine multiple sampling methods
    # in a sequence. Each sampler in the chain filters the token probabilities
    # before passing them to the next sampler.
    #
    # Example:
    # ```
    # chain = Llama::Sampler::Chain.new
    # chain.add(Llama::Sampler::TopK.new(40))      # First filter with top-k
    # chain.add(Llama::Sampler::TopP.new(0.95, 1)) # Then apply top-p
    # chain.add(Llama::Sampler::Temp.new(0.8))     # Apply temperature
    # chain.add(Llama::Sampler::Dist.new(42))      # Final distribution sampling
    #
    # # Generate text with the custom sampler chain
    # result = context.generate_with_sampler("Your prompt", chain, 150)
    # ```
    class Chain < Base
      # Creates a new SamplerChain with optional parameters
      #
      # Parameters:
      # - no_perf: Whether to disable performance counters (default: false)
      #
      # Raises:
      # - Llama::Error if the sampler chain cannot be created
      def initialize(no_perf : Bool = false)
        params = LibLlama.llama_sampler_chain_default_params
        params.no_perf = no_perf
        @handle = LibLlama.llama_sampler_chain_init(params)
        raise Error.new("Failed to create sampler chain") if @handle.null?
        @samplers = [] of Base
      end

      # Adds a sampler to the chain
      #
      # Parameters:
      # - sampler: The sampler to add to the chain
      def add(sampler : Base)
        LibLlama.llama_sampler_chain_add(@handle, sampler.to_unsafe)
        @samplers << sampler # Keep reference to prevent GC
      end

      # Samples a token using the sampler chain
      #
      # Parameters:
      # - ctx: The context to sample from
      # - idx: The index of the logits to sample from (-1 for the last token)
      #
      # Returns:
      # - The sampled token
      def sample(ctx : Context, idx : Int32 = -1) : Int32
        LibLlama.llama_sampler_sample(@handle, ctx.to_unsafe, idx)
      end

      # Accepts a token, updating the internal state of the samplers
      #
      # Parameters:
      # - token: The token to accept
      def accept(token : Int32)
        LibLlama.llama_sampler_accept(@handle, token)
      end

      # Frees the resources associated with this sampler chain
      # Overrides the parent class's finalize method to also clean up the samplers array
      def finalize
        # Clear the samplers array to allow GC to collect the samplers
        @samplers.clear if @samplers
        super
      end

      # Print performance information for this sampler chain
      #
      # This method prints performance statistics about the sampler chain to STDERR.
      # It's useful for debugging and performance analysis.
      def print_perf
        LibLlama.llama_perf_sampler_print(@handle)
      end

      # Reset performance counters for this sampler chain
      #
      # This method resets all performance counters for the sampler chain.
      def reset_perf
        LibLlama.llama_perf_sampler_reset(@handle)
      end

      @samplers : Array(Base)
    end
  end
end
