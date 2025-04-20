module Llama
  module Sampler
    # Distribution sampler (final sampler in a chain)
    #
    # The Distribution sampler is typically the final sampler in a chain.
    # It samples a token from the probability distribution after all other
    # samplers have filtered the logits. This sampler is required to actually
    # select a token from the distribution.
    #
    # Example:
    # ```
    # sampler = Llama::Sampler::Dist.new(42) # Use seed 42 for reproducibility
    # ```
    class Dist < Base
      # Creates a new distribution sampler
      #
      # Parameters:
      # - seed: Random seed for sampling (default: LLAMA_DEFAULT_SEED)
      #
      # Raises:
      # - Llama::Error if the sampler cannot be created
      def initialize(seed : UInt32 = LibLlama::LLAMA_DEFAULT_SEED)
        handle = LibLlama.llama_sampler_init_dist(seed)
        raise Error.new("Failed to create distribution sampler") if handle.null?
        super(handle)
      end
    end
  end
end
