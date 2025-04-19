module Llama
  module Sampler
    # Penalties sampler
    #
    # The Penalties sampler applies various penalties to token probabilities
    # to reduce repetition and improve diversity in the generated text.
    # It can penalize recently used tokens, frequent tokens, and more.
    #
    # Example:
    # ```
    # # Apply penalties to the last 64 tokens with a repetition penalty of 1.1
    # sampler = Llama::Sampler::Penalties.new(64, 1.1, 0.0, 0.0)
    # ```
    class Penalties < Base
      # Creates a new Penalties sampler
      #
      # Parameters:
      # - penalty_last_n: Last n tokens to penalize (0 = disable, -1 = context size)
      # - penalty_repeat: Repetition penalty (1.0 = disabled)
      # - penalty_freq: Frequency penalty (0.0 = disabled)
      # - penalty_present: Presence penalty (0.0 = disabled)
      #
      # Raises:
      # - Llama::Error if the sampler cannot be created
      def initialize(penalty_last_n : Int32, penalty_repeat : Float32, penalty_freq : Float32, penalty_present : Float32)
        @handle = LibLlama.llama_sampler_init_penalties(
          penalty_last_n,
          penalty_repeat,
          penalty_freq,
          penalty_present
        )
        raise Error.new("Failed to create Penalties sampler") if @handle.null?
      end
    end
  end
end
