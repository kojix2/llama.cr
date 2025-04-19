module Llama
  module Sampler
    # Mirostat sampler (version 2)
    #
    # The Mirostat V2 sampler is an improved version of the Mirostat algorithm
    # that requires fewer parameters and is more efficient. It dynamically
    # adjusts sampling to maintain a target entropy level.
    #
    # Based on the paper: https://arxiv.org/abs/2007.14966
    #
    # Example:
    # ```
    # sampler = Llama::Sampler::MirostatV2.new(42, 5.0, 0.1)
    # ```
    class MirostatV2 < Base
      # Creates a new Mirostat V2 sampler
      #
      # Parameters:
      # - seed: Random seed
      # - tau: Target entropy (5.0 is a good default)
      # - eta: Learning rate (0.1 is a good default)
      #
      # Raises:
      # - Llama::Error if the sampler cannot be created
      def initialize(seed : UInt32, tau : Float32, eta : Float32)
        @handle = LibLlama.llama_sampler_init_mirostat_v2(seed, tau, eta)
        raise Error.new("Failed to create Mirostat V2 sampler") if @handle.null?
      end
    end
  end
end
