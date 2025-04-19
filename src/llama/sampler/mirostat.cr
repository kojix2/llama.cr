module Llama
  module Sampler
    # Mirostat sampler (version 1)
    #
    # The Mirostat sampler dynamically adjusts the temperature to maintain
    # a target entropy level in the generated text. This helps to produce
    # consistent quality output regardless of the context.
    #
    # Based on the paper: https://arxiv.org/abs/2007.14966
    #
    # Example:
    # ```
    # sampler = Llama::Sampler::Mirostat.new(32000, 42, 5.0, 0.1, 100)
    # ```
    class Mirostat < Base
      # Creates a new Mirostat sampler
      #
      # Parameters:
      # - n_vocab: Vocabulary size
      # - seed: Random seed
      # - tau: Target entropy (5.0 - 8.0 is a good range)
      # - eta: Learning rate (0.1 is a good default)
      # - m: Number of tokens for estimating entropy (100 is a good default)
      #
      # Raises:
      # - Llama::Error if the sampler cannot be created
      def initialize(n_vocab : Int32, seed : UInt32, tau : Float32, eta : Float32, m : Int32)
        @handle = LibLlama.llama_sampler_init_mirostat(n_vocab, seed, tau, eta, m)
        raise Error.new("Failed to create Mirostat sampler") if @handle.null?
      end
    end
  end
end
