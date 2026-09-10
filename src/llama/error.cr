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

  # Raised when an operation uses a native resource after it has been closed.
  class ClosedError < Error
    getter resource_type : String

    def initialize(@resource_type : String)
      super("#{@resource_type} is closed")
    end
  end

  # Raised when closing or mutating a resource used by an active operation.
  class BusyError < Error
    getter resource_type : String

    def initialize(@resource_type : String)
      super("#{@resource_type} is busy")
    end
  end

  class UnsupportedOperationError < Error
  end

  # Raised by checked decode operations when llama.cpp does not accept a batch.
  #
  # `Context#decode` retains its compatibility behavior, including returning
  # positive native status codes. `Context#decode!` raises this error for every
  # non-zero native status so higher-level code cannot continue with stale
  # logits.
  class DecodeError < Error
    enum Reason
      NoKvSlot
      Aborted
      InvalidBatch
      Fatal
    end

    getter operation : String
    getter native_code : Int32
    getter reason : Reason
    getter batch_size : Int32

    def initialize(@operation : String, @native_code : Int32, @reason : Reason, @batch_size : Int32)
      super(Llama.format_error(
        "#{@operation} failed",
        @native_code,
        "reason: #{@reason}, batch size: #{@batch_size}"
      ))
    end

    def self.reason_for(code : Int32) : Reason
      case code
      when  1 then Reason::NoKvSlot
      when  2 then Reason::Aborted
      when -1 then Reason::InvalidBatch
      else         Reason::Fatal
      end
    end
  end
end
