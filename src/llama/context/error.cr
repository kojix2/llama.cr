require "../error"

module Llama
  class Context
    class Error < Llama::Error; end
  end

  # A context-capacity error that remains compatible with Context::Error.
  class ContextFullError < Context::Error; end
end
