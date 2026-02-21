module Llama
  module Sampler
    # Adaptive-P sampler
    #
    # Selects tokens near a configurable target probability over time.
    class AdaptiveP < Base
      # Creates a new Adaptive-P sampler.
      #
      # Parameters:
      # - target: Target probability (0.0 to 1.0, negative to disable)
      # - decay: EMA decay (0.0 to 0.99)
      # - seed: RNG seed
      #
      # Raises:
      # - Llama::Error if the sampler cannot be created
      def initialize(target : Float32, decay : Float32, seed : UInt32 = LibLlama::LLAMA_DEFAULT_SEED)
        handle = LibLlama.llama_sampler_init_adaptive_p(target, decay, seed)
        raise Error.new("Failed to create adaptive-p sampler") if handle.null?
        super(handle)
      end
    end
  end
end
