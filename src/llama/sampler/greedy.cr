module Llama
  module Sampler
    # The Greedy sampler always selects the token with the highest probability.
    # This is the simplest sampling method and produces deterministic output.
    #
    # Example:
    # ```
    # sampler = Llama::Sampler::Greedy.new
    # ```
    class Greedy < Base
      # Creates a new greedy sampler
      #
      # Raises:
      # - Llama::Error if the sampler cannot be created
      def initialize
        handle = LibLlama.llama_sampler_init_greedy
        raise Error.new("Failed to create greedy sampler") if handle.null?
        super(handle)
      end
    end
  end
end
