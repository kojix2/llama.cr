module Llama
  module Sampler
    # Top-K sampler
    #
    # The Top-K sampler keeps only the K most likely tokens and zeros out the rest.
    # This is a simple but effective way to filter out unlikely tokens.
    #
    # Example:
    # ```
    # sampler = Llama::Sampler::TopK.new(40) # Keep only the top 40 tokens
    # ```
    class TopK < Base
      # Creates a new Top-K sampler
      #
      # Parameters:
      # - k: The number of top tokens to consider
      #
      # Raises:
      # - Llama::Error if the sampler cannot be created
      def initialize(k : Int32)
        @handle = LibLlama.llama_sampler_init_top_k(k)
        raise Error.new("Failed to create Top-K sampler") if @handle.null?
      end
    end
  end
end
