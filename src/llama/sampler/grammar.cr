module Llama
  module Sampler
    # Grammar sampler
    #
    # The Grammar sampler constrains token generation to follow a formal grammar
    # defined in GBNF format. This is useful for generating structured text like
    # JSON, XML, or code that must follow specific syntax rules.
    #
    # Example:
    # ```
    # # Define a simple grammar for a numbered list
    # grammar = %q{
    #   root ::= list
    #   list ::= item+
    #   item ::= number " " text "\n"
    #   number ::= "1" | "2" | "3" | "4" | "5"
    #   text ::= [a-zA-Z ]+
    # }
    #
    # sampler = Llama::Sampler::Grammar.new(model.vocab, grammar, "root")
    # ```
    class Grammar < Base
      # Creates a new Grammar sampler
      #
      # Parameters:
      # - vocab: The vocabulary to use
      # - grammar_str: The grammar definition string in GBNF format
      # - grammar_root: The root symbol of the grammar
      #
      # Raises:
      # - Llama::Error if the sampler cannot be created
      def initialize(vocab : Vocab, grammar_str : String, grammar_root : String)
        handle = LibLlama.llama_sampler_init_grammar(
          vocab.to_unsafe,
          grammar_str,
          grammar_root
        )
        raise Error.new("Failed to create Grammar sampler") if handle.null?
        super(handle)
        # Store references to prevent GC
        @vocab = vocab
        @grammar_str = grammar_str
        @grammar_root = grammar_root
      end

      # Overrides the parent class's finalize method to ensure proper cleanup
      def finalize
        # First nullify our references to prevent circular references
        @vocab = nil
        @grammar_str = nil
        @grammar_root = nil

        # Then call the parent's finalize method
        super
      end

      # Instance variables to keep references to prevent GC
      @vocab : Vocab?
      @grammar_str : String?
      @grammar_root : String?
    end
  end
end
