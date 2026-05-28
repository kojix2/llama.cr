# src/llama/chat.cr
# Production-grade chat template handling with memory safety fixes

require "log"

module Llama
  Log = ::Log.for("llama.chat")

  # Represents a message in a chat conversation
  class ChatMessage
    property role : String
    property content : String

    def initialize(@role : String, @content : String)
    end

    # Creates a C struct with pointers to the message data
    def to_unsafe : LibLlama::LlamaChatMessage
      msg = LibLlama::LlamaChatMessage.new
      msg.role = @role.to_unsafe
      msg.content = @content.to_unsafe
      msg
    end
  end

  # Applies a chat template to a list of messages
  #
  # Parameters:
  # - template: The template string (nil to use model's default)
  # - messages: Array of chat messages
  # - add_assistant: Whether to end with an assistant message prefix
  #
  # Returns:
  # - The formatted prompt string
  #
  # Raises:
  # - ArgumentError if messages array is empty
  # - Llama::Error if template application fails
  def self.apply_chat_template(
    template : String?,
    messages : Array(ChatMessage),
    add_assistant : Bool = true,
  ) : String
    if messages.empty?
      raise ArgumentError.new("messages array cannot be empty")
    end

    Log.debug { "Applying chat template to #{messages.size} messages, add_assistant=#{add_assistant}" }

    tmpl = template || ""

    # Build C messages array - the Crystal strings must remain alive during this call
    c_messages = messages.map(&.to_unsafe)

    # First call: get required buffer size
    required_size = LibLlama.llama_chat_apply_template(
      tmpl.to_unsafe,
      c_messages.to_unsafe,
      messages.size,
      add_assistant,
      nil,
      0
    )

    if required_size < 0
      error_msg = Llama.format_error(
        "Failed to apply chat template",
        required_size,
        "template: #{tmpl.size} chars, messages: #{messages.size}"
      )
      raise Error.new(error_msg)
    end

    if required_size == 0
      raise Error.new("Chat template returned empty result - template may be invalid")
    end

    # Allocate buffer with null check
    buffer = Pointer(LibC::Char).malloc(required_size)
    if buffer.null?
      raise Error.new("Failed to allocate #{required_size} bytes for chat template output")
    end

    begin
      written = LibLlama.llama_chat_apply_template(
        tmpl.to_unsafe,
        c_messages.to_unsafe,
        messages.size,
        add_assistant,
        buffer,
        required_size
      )

      if written < 0
        error_msg = Llama.format_error(
          "Failed to apply chat template on second call",
          written,
          nil
        )
        raise Error.new(error_msg)
      end

      if written > required_size
        raise Error.new("Chat template output (#{written}) exceeded allocated buffer (#{required_size})")
      end

      # Convert result to string
      String.new(buffer, written)
    ensure
      # Free the allocated buffer
      LibC.free(buffer)
    end
  end

  # Gets the list of built-in chat templates
  #
  # Returns:
  # - Array of template names
  def self.builtin_chat_templates : Array(String)
    # First call: get required count
    count = LibLlama.llama_chat_builtin_templates(Pointer(LibC::Char*).null, 0)
    
    if count <= 0
      return [] of String
    end

    # Allocate exactly enough space
    output = Pointer(LibC::Char*).malloc(count)
    if output.null?
      raise Error.new("Failed to allocate memory for template list")
    end

    begin
      actual_count = LibLlama.llama_chat_builtin_templates(output, count)
      
      if actual_count <= 0
        return [] of String
      end

      result = Array(String).new(actual_count)
      actual_count.times do |i|
        if output[i] && !output[i].null?
          result << String.new(output[i])
        end
      end
      result
    ensure
      LibC.free(output)
    end
  end
end
