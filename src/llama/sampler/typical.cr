module Llama
  module Sampler
    # Typical sampler
    #
    # The Typical sampler selects tokens based on their "typicality" (entropy).
    # It filters out tokens that are either too predictable or too surprising,
    # leading to more natural and diverse text generation.
    #
    # Based on the paper: https://arxiv.org/abs/2202.00666
    #
    # Example:
    # ```
    # sampler = Llama::Sampler::Typical.new(0.95, 1) # Keep tokens with typicality >= 0.95
    # ```
    class Typical < Base
      # Creates a new Typical sampler
      #
      # Parameters:
      # - p: The typicality threshold (0.0 to 1.0)
      # - min_keep: Minimum number of tokens to keep
      #
      # Raises:
      # - Llama::Error if the sampler cannot be created
      def initialize(p : Float32, min_keep : Int32 = 1)
        @handle = LibLlama.llama_sampler_init_typical(p, min_keep)
        raise Error.new("Failed to create Typical sampler") if @handle.null?
      end
    end
  end
end
