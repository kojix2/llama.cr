module Llama
  module Sampler
    # Top-P (nucleus) sampler
    #
    # The Top-P sampler (also known as nucleus sampling) keeps tokens whose
    # cumulative probability exceeds the threshold P. This helps to filter out
    # the long tail of low-probability tokens.
    #
    # Example:
    # ```
    # sampler = Llama::Sampler::TopP.new(0.95, 1) # Keep tokens until 95% probability is covered
    # ```
    class TopP < Base
      # Creates a new Top-P sampler
      #
      # Parameters:
      # - p: The cumulative probability threshold (0.0 to 1.0)
      # - min_keep: Minimum number of tokens to keep
      #
      # Raises:
      # - Llama::Error if the sampler cannot be created
      def initialize(p : Float32, min_keep : Int32 = 1)
        @handle = LibLlama.llama_sampler_init_top_p(p, min_keep)
        raise Error.new("Failed to create Top-P sampler") if @handle.null?
      end
    end
  end
end
