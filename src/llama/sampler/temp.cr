module Llama
  module Sampler
    # Temperature sampler
    #
    # The Temperature sampler adjusts the logits by dividing them by the temperature
    # value. Higher temperatures (>1.0) make the distribution more uniform, leading
    # to more random outputs. Lower temperatures (<1.0) make the distribution more
    # peaked, leading to more deterministic outputs.
    #
    # Example:
    # ```
    # sampler = Llama::Sampler::Temp.new(0.8) # Slightly more deterministic than default
    # ```
    class Temp < Base
      # Creates a new temperature sampler
      #
      # Parameters:
      # - temp: The temperature value (0.0 = greedy, 1.0 = normal, >1.0 = more random)
      #
      # Raises:
      # - Llama::Error if the sampler cannot be created
      def initialize(temp : Float32)
        @handle = LibLlama.llama_sampler_init_temp(temp)
        raise Error.new("Failed to create temperature sampler") if @handle.null?
      end
    end
  end
end
