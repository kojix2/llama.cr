module Llama
  module Sampler
    # Top-N Sigma sampler
    #
    # The Top-N Sigma sampler selects tokens based on their distance from the mean
    # in terms of standard deviations. This is based on the paper "Top-nσ: Not All Logits Are You Need"
    # (https://arxiv.org/pdf/2411.07641).
    #
    # Example:
    # ```
    # sampler = Llama::Sampler::TopNSigma.new(2.0) # Keep tokens within 2 standard deviations
    # ```
    class TopNSigma < Base
      # Creates a new Top-N Sigma sampler
      #
      # Parameters:
      # - n: Number of standard deviations to keep
      #
      # Raises:
      # - Llama::Error if the sampler cannot be created
      def initialize(n : Float32)
        @handle = LibLlama.llama_sampler_init_top_n_sigma(n)
        raise Error.new("Failed to create Top-N Sigma sampler") if @handle.null?
      end
    end
  end
end
