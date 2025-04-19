module Llama
  module Sampler
    # Grammar Lazy Patterns sampler
    #
    # The Grammar Lazy Patterns sampler is an extension of the Grammar sampler
    # that only applies grammar constraints when triggered by specific patterns
    # or tokens. This is useful for mixed-format generation where grammar
    # constraints should only apply to certain parts of the output.
    #
    # Example:
    # ```
    # # Define a JSON grammar that only activates when the text contains "JSON:"
    # grammar = %q{
    #   root ::= object
    #   object ::= "{" ws (string ":" ws value ("," ws string ":" ws value)*)? "}" ws
    #   array ::= "[" ws (value ("," ws value)*)? "]" ws
    #   value ::= object | array | string | number | "true" | "false" | "null"
    #   string ::= "\"" ([^"\\] | "\\" .)* "\""
    #   number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?
    #   ws ::= [ \t\n]*
    # }
    #
    # trigger_patterns = ["JSON:"]
    # sampler = Llama::Sampler::GrammarLazyPatterns.new(
    #   model.vocab, grammar, "root", trigger_patterns
    # )
    # ```
    class GrammarLazyPatterns < Base
      # Creates a new Grammar Lazy Patterns sampler
      #
      # Parameters:
      # - vocab: The vocabulary to use
      # - grammar_str: The grammar definition string in GBNF format
      # - grammar_root: The root symbol of the grammar
      # - trigger_patterns: Array of string patterns that will trigger the grammar
      # - trigger_tokens: Array of token IDs that will trigger the grammar
      #
      # Raises:
      # - Llama::Error if the sampler cannot be created
      def initialize(
        vocab : Vocab,
        grammar_str : String,
        grammar_root : String,
        trigger_patterns : Array(String) = [] of String,
        trigger_tokens : Array(Int32) = [] of Int32,
      )
        # Convert trigger patterns to C-style array
        patterns_ptr = Pointer(LibC::Char*).malloc(trigger_patterns.size + 1)
        trigger_patterns.each_with_index do |pattern, i|
          patterns_ptr[i] = pattern.to_unsafe
        end
        patterns_ptr[trigger_patterns.size] = Pointer(LibC::Char).null

        # Convert trigger tokens to C-style array
        tokens_ptr = Pointer(LibLlama::LlamaToken).null
        if trigger_tokens.size > 0
          tokens_ptr = Pointer(LibLlama::LlamaToken).malloc(trigger_tokens.size)
          trigger_tokens.each_with_index do |token, i|
            tokens_ptr[i] = token
          end
        end

        @handle = LibLlama.llama_sampler_init_grammar_lazy_patterns(
          vocab.to_unsafe,
          grammar_str,
          grammar_root,
          patterns_ptr,
          trigger_patterns.size,
          tokens_ptr,
          trigger_tokens.size
        )
        raise Error.new("Failed to create Grammar Lazy Patterns sampler") if @handle.null?

        # Store references to prevent GC
        @vocab = vocab
        @grammar_str = grammar_str
        @grammar_root = grammar_root
        @trigger_patterns = trigger_patterns
        @trigger_tokens = trigger_tokens
      end

      # Overrides the parent class's finalize method to ensure proper cleanup
      def finalize
        # First nullify our references to prevent circular references
        @vocab = nil
        @grammar_str = nil
        @grammar_root = nil
        @trigger_patterns = nil
        @trigger_tokens = nil

        # Then call the parent's finalize method
        super
      end

      # Instance variables to keep references to prevent GC
      @vocab : Vocab?
      @grammar_str : String?
      @grammar_root : String?
      @trigger_patterns : Array(String)?
      @trigger_tokens : Array(Int32)?
    end
  end
end
