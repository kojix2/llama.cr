module Llama
  module Sampler
    # XTC sampler
    #
    # The XTC sampler combines aspects of several sampling methods for improved
    # text generation quality. It was introduced in the Text Generation WebUI project.
    #
    # Example:
    # ```
    # sampler = Llama::Sampler::Xtc.new(0.3, 0.8, 1, 42)
    # ```
    class Xtc < Base
      # Creates a new XTC sampler
      #
      # Parameters:
      # - p: Probability threshold
      # - t: Temperature value
      # - min_keep: Minimum number of tokens to keep
      # - seed: Random seed for sampling
      #
      # Raises:
      # - Llama::Error if the sampler cannot be created
      def initialize(p : Float32, t : Float32, min_keep : Int32, seed : UInt32 = LibLlama::LLAMA_DEFAULT_SEED)
        @handle = LibLlama.llama_sampler_init_xtc(p, t, min_keep, seed)
        raise Error.new("Failed to create XTC sampler") if @handle.null?
      end
    end
  end
end
