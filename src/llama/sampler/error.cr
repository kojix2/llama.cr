require "../error"

module Llama
  module Sampler
    class Error < Llama::Error; end

    class TokenizationError < Error; end
  end
end
