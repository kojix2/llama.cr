module Llama
  module Sampler
    # Min-P sampler
    #
    # The Min-P sampler keeps tokens with probability >= P * max_probability.
    # This is similar to Top-P but uses a minimum probability threshold relative
    # to the most likely token, rather than a cumulative probability threshold.
    #
    # Example:
    # ```
    # sampler = Llama::Sampler::MinP.new(0.05, 1) # Keep tokens with prob >= 5% of max prob
    # ```
    class MinP < Base
      # Creates a new Min-P sampler
      #
      # Parameters:
      # - p: The minimum probability threshold (0.0 to 1.0)
      # - min_keep: Minimum number of tokens to keep
      #
      # Raises:
      # - Llama::Error if the sampler cannot be created
      def initialize(p : Float32, min_keep : Int32 = 1)
        handle = LibLlama.llama_sampler_init_min_p(p, min_keep)
        raise Error.new("Failed to create min-p sampler") if handle.null?
        super(handle)
      end
    end
  end
end
