module Llama
  # Represents a message in a chat conversation
  class ChatMessage
    # The role of the message sender (e.g., "system", "user", "assistant")
    property role : String

    # The content of the message
    property content : String

    # Creates a new ChatMessage
    #
    # Parameters:
    # - role: The role of the message sender
    # - content: The content of the message
    def initialize(@role : String, @content : String)
    end

    # Converts to the C structure
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
  # - Llama::Error if template application fails
  def self.apply_chat_template(
    template : String?,
    messages : Array(ChatMessage),
    add_assistant : Bool = true,
  ) : String
    # Convert messages to C structures
    c_messages = messages.map(&.to_unsafe)

    # Estimate buffer size
    estimated_size = messages.sum { |msg| msg.content.size + msg.role.size } * 2
    buffer = Pointer(LibC::Char).malloc(estimated_size)

    # Apply template
    result = LibLlama.llama_chat_apply_template(
      (template || "").to_unsafe,
      c_messages.to_unsafe,
      messages.size,
      add_assistant,
      buffer,
      estimated_size
    )

    # Retry with larger buffer if needed
    if result > estimated_size
      buffer = Pointer(LibC::Char).malloc(result)
      result = LibLlama.llama_chat_apply_template(
        (template || "").to_unsafe,
        c_messages.to_unsafe,
        messages.size,
        add_assistant,
        buffer,
        result
      )
    end

    # Check for errors
    raise Error.new("Failed to apply chat template") if result < 0

    # Convert result to string
    String.new(buffer, result)
  end

  # Gets the list of built-in chat templates
  #
  # Returns:
  # - Array of template names
  def self.builtin_chat_templates : Array(String)
    # Assume maximum of 100 templates
    output = Pointer(LibC::Char*).malloc(100)
    count = LibLlama.llama_chat_builtin_templates(output, 100)

    result = [] of String
    count.times do |i|
      result << String.new(output[i])
    end

    result
  end
end
