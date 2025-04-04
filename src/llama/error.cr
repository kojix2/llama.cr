module Llama
  # Base error class for all llama.cr errors
  class Error < Exception; end

  # Errors related to model loading and operations
  class ModelError < Error; end

  # Errors related to context operations
  class ContextError < Error; end

  # Errors related to batch processing
  class BatchError < Error; end

  # Errors related to tokenization
  class TokenizationError < Error; end

  # Errors related to KV cache operations
  class KvCacheError < Error; end

  # Errors related to state management
  class StateError < Error; end

  # Errors related to sampling operations
  class SamplingError < Error; end

  # Mapping of error codes to error messages
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

  # Get error message from error code
  def self.error_message(code : Int32) : String
    ERROR_MESSAGES[code]? || "Unknown error (code: #{code})"
  end

  # Format error message with additional context
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
