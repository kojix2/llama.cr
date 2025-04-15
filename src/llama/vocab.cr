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

    # Converts a token to a piece of text
    # This is similar to token_to_text but provides more control over the output format
    #
    # Parameters:
    # - token: The token to convert
    # - lstrip: Whether to strip leading spaces (0 = no, 1 = yes)
    # - special: Whether to render special tokens
    #
    # Returns:
    # - The text representation of the token
    def token_to_piece(token : Int32, lstrip : Int32 = 0, special : Bool = false) : String
      buf_size = 128
      buf = Pointer(LibC::Char).malloc(buf_size)

      n = LibLlama.llama_token_to_piece(@handle, token, buf, buf_size, lstrip, special)

      if n < 0
        raise Error.new("Failed to convert token to piece")
      end

      String.new(buf, n)
    end

    # Format a token for display
    #
    # Parameters:
    # - token: The token to format
    # - show_id: Whether to show the token ID
    # - show_text: Whether to show the token text
    #
    # Returns:
    # - A formatted string representation of the token
    def format_token(token : Int32, show_id : Bool = true, show_text : Bool = true) : String
      if show_id && show_text
        piece = token_to_piece(token, 0, true)
        "%6d -> '#{piece}'" % token
      elsif show_id
        token.to_s
      elsif show_text
        token_to_piece(token, 0, true)
      else
        ""
      end
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

    # Returns whether the model adds BOS token by default
    def add_bos? : Bool
      LibLlama.llama_vocab_get_add_bos(@handle)
    end

    # Returns whether the model adds EOS token by default
    def add_eos? : Bool
      LibLlama.llama_vocab_get_add_eos(@handle)
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

    # Checks if a token is an end-of-generation token
    def is_eog(token : Int32) : Bool
      LibLlama.llama_vocab_is_eog(@handle, token)
    end

    # Checks if a token is a control token
    def is_control(token : Int32) : Bool
      LibLlama.llama_vocab_is_control(@handle, token)
    end

    # Returns the raw pointer to the underlying llama_vocab structure
    def to_unsafe
      @handle
    end

    @handle : LibLlama::LlamaVocab*
  end
end
