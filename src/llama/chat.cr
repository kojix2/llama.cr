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

    tmpl = template || ""

    # First call: get required buffer size
    required_size = LibLlama.llama_chat_apply_template(
      tmpl.to_unsafe,
      c_messages.to_unsafe,
      messages.size,
      add_assistant,
      nil,
      0
    )

    # Check for errors
    raise Error.new("Failed to apply chat template") if required_size < 0

    # Second call: allocate buffer and get the result
    buffer = Pointer(LibC::Char).malloc(required_size)
    begin
      written = LibLlama.llama_chat_apply_template(
        tmpl.to_unsafe,
        c_messages.to_unsafe,
        messages.size,
        add_assistant,
        buffer,
        required_size
      )

      # Check for errors
      raise Error.new("Failed to apply chat template") if written < 0
      raise Error.new("Chat template output exceeded allocated buffer") if written > required_size

      # Convert result to string
      String.new(buffer, written)
    ensure
      LibC.free(buffer)
    end
  end

  # Gets the list of built-in chat templates
  #
  # Returns:
  # - Array of template names
  def self.builtin_chat_templates : Array(String)
    capacity = 100
    output = Pointer(LibC::Char*).malloc(capacity)

    begin
      count = LibLlama.llama_chat_builtin_templates(output, capacity)

      if count > capacity
        LibC.free(output)
        capacity = count
        output = Pointer(LibC::Char*).malloc(capacity)
        count = LibLlama.llama_chat_builtin_templates(output, capacity)
      end

      result = [] of String
      count.times do |i|
        result << String.new(output[i])
      end

      result
    ensure
      LibC.free(output)
    end
  end
end
