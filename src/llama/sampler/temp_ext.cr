module Llama
  module Sampler
    # Extended Temperature sampler
    #
    # The Extended Temperature sampler provides more control over temperature
    # sampling with additional parameters for dynamic temperature adjustment.
    # This is based on the paper "Dynamic Temperature for Language Models" (https://arxiv.org/abs/2309.02772).
    #
    # Example:
    # ```
    # # Create a temperature sampler with base temp 0.8, delta 0.5, and exponent 1.0
    # sampler = Llama::Sampler::TempExt.new(0.8, 0.5, 1.0)
    # ```
    class TempExt < Base
      # Creates a new Extended Temperature sampler
      #
      # Parameters:
      # - t: Base temperature value
      # - delta: Temperature delta for dynamic adjustment
      # - exponent: Exponent for the temperature formula
      #
      # Raises:
      # - Llama::Error if the sampler cannot be created
      def initialize(t : Float32, delta : Float32, exponent : Float32)
        handle = LibLlama.llama_sampler_init_temp_ext(t, delta, exponent)
        raise Error.new("Failed to create temp-ext sampler") if handle.null?
        super(handle)
      end
    end
  end
end
