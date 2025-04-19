require "../error"

module Llama
  class Context
    class Error < Llama::Error; end

    class TokenizationError < Llama::Error; end
  end
end
