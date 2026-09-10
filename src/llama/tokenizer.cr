module Llama
  # Safe high-level facade over a model vocabulary.
  class Tokenizer
    def initialize(@vocab : Vocab)
    end

    def encode(text : String, add_special : Bool = true, parse_special : Bool = true) : Array(Token)
      @vocab.tokenize(text, add_special, parse_special)
    end

    def decode(tokens : Array(Token), remove_special : Bool = true, unparse_special : Bool = false) : String
      @vocab.detokenize(tokens, remove_special, unparse_special)
    end

    def piece(token : Token, lstrip : Int32 = 0, special : Bool = false) : Bytes
      @vocab.token_to_piece_bytes(token, lstrip, special)
    end
  end
end
