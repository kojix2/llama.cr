module Llama
  module Sampler
    # Infill sampler
    #
    # The Infill sampler is designed for fill-in-the-middle (FIM) tasks.
    # It helps to generate text that fits naturally between existing content.
    #
    # Example:
    # ```
    # sampler = Llama::Sampler::Infill.new(model.vocab)
    # ```
    class Infill < Base
      # Creates a new Infill sampler
      #
      # Parameters:
      # - vocab: The vocabulary to use
      #
      # Raises:
      # - Llama::Error if the sampler cannot be created
      def initialize(vocab : Vocab)
        handle = LibLlama.llama_sampler_init_infill(vocab.to_unsafe)
        raise Error.new("Failed to create Infill sampler") if handle.null?
        super(handle)
        # Store reference to prevent GC
        @vocab = vocab
      end

      # Overrides the parent class's finalize method to ensure proper cleanup
      def finalize
        # First nullify our reference to prevent circular references
        @vocab = nil

        # Then call the parent's finalize method
        super
      end

      # Instance variable to keep reference to prevent GC
      @vocab : Vocab?
    end
  end
end
