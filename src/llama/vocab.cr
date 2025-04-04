module Llama
  # Wrapper for the llama_vocab structure
  class Vocab
    # Creates a new Vocab instance from a raw pointer
    #
    # Note: This constructor is intended for internal use.
    # Users should obtain Vocab instances through Model#vocab.
    def initialize(@handle : LibLlama::LlamaVocab*)
    end

    # Returns the number of tokens in the vocabulary
    def n_tokens : Int32
      LibLlama.llama_vocab_n_tokens(@handle)
    end

    # Returns the text representation of a token
    def token_to_text(token : Int32) : String
      ptr = LibLlama.llama_vocab_get_text(@handle, token)
      String.new(ptr)
    end

    # Tokenizes a string into an array of token IDs
    def tokenize(text : String, add_special : Bool = true, parse_special : Bool = true) : Array(Int32)
      max_tokens = text.size * 2 # A reasonable upper bound
      tokens = Pointer(LibLlama::LlamaToken).malloc(max_tokens)

      n_tokens = LibLlama.llama_tokenize(
        @handle,
        text,
        text.bytesize,
        tokens,
        max_tokens,
        add_special,
        parse_special
      )

      if n_tokens < 0
        # If n_tokens is negative, it indicates the required buffer size
        max_tokens = -n_tokens
        tokens = Pointer(LibLlama::LlamaToken).malloc(max_tokens)

        n_tokens = LibLlama.llama_tokenize(
          @handle,
          text,
          text.bytesize,
          tokens,
          max_tokens,
          add_special,
          parse_special
        )
      end

      raise Error.new("Failed to tokenize text") if n_tokens < 0

      result = Array(Int32).new(n_tokens)
      n_tokens.times do |i|
        result << tokens[i]
      end

      result
    end

    # Special token methods

    # Returns the beginning-of-sentence token ID
    def bos : Int32
      LibLlama.llama_vocab_bos(@handle)
    end

    # Returns the end-of-sentence token ID
    def eos : Int32
      LibLlama.llama_vocab_eos(@handle)
    end

    # Returns the end-of-turn token ID
    def eot : Int32
      LibLlama.llama_vocab_eot(@handle)
    end

    # Returns the newline token ID
    def nl : Int32
      LibLlama.llama_vocab_nl(@handle)
    end

    # Returns the padding token ID
    def pad : Int32
      LibLlama.llama_vocab_pad(@handle)
    end

    # Returns the raw pointer to the underlying llama_vocab structure
    def to_unsafe
      @handle
    end

    @handle : LibLlama::LlamaVocab*
  end
end
