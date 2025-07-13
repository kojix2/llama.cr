module Llama
  class Error < Exception
    ERROR_MESSAGES = {
       -1 => "General error",
       -2 => "Memory allocation error",
       -3 => "Batch processing error",
       -4 => "Context creation error",
       -5 => "Model loading error",
       -6 => "Tokenization error",
       -7 => "KV cache error",
       -8 => "State management error",
       -9 => "Sampling error",
      -10 => "Invalid parameter error",
      -11 => "File I/O error",
      -12 => "Network error",
      -13 => "GPU error",
      -14 => "Timeout error",
      -15 => "Unsupported operation error",
    }

    def self.error_message(code : Int32) : String
      ERROR_MESSAGES[code]? || "Unknown error (code: #{code})"
    end

    def self.format_error(message : String, code : Int32? = nil, context : String? = nil) : String
      result = message
      if code
        error_msg = error_message(code)
        result += " - #{error_msg} (code: #{code})"
      end
      if context
        result += " [#{context}]"
      end
      result
    end
  end

  def self.error_message(code : Int32) : String
    Error.error_message(code)
  end

  def self.format_error(message : String, code : Int32? = nil, context : String? = nil) : String
    Error.format_error(message, code, context)
  end

  # Specific error class for tokenization errors
  class TokenizationError < Error
  end
end
