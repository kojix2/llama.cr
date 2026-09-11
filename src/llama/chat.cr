module Llama
  private MAX_CHAT_TEMPLATE_BYTES = 64 * 1024 * 1024

  record Message, role : String, content : String do
    def self.system(content : String) : self
      new("system", content)
    end

    def self.user(content : String) : self
      new("user", content)
    end

    def self.assistant(content : String) : self
      new("assistant", content)
    end

    def self.tool(content : String) : self
      new("tool", content)
    end

    def to_chat_message : ChatMessage
      ChatMessage.new(role, content)
    end
  end

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
  # - TemplateError if the template is unsupported or formatting fails
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

    # b10809 recognizes a predefined set of Jinja template shapes; it is not a
    # general Jinja evaluator. Surface an unsupported custom template distinctly.
    raise TemplateError.new("chat template is not recognized by llama.cpp") if required_size < 0
    if required_size > MAX_CHAT_TEMPLATE_BYTES
      raise TemplateError.new("chat template output exceeds the safety limit")
    end

    # Second call: allocate buffer and get the result
    buffer = Bytes.new(required_size)
    written = LibLlama.llama_chat_apply_template(
      tmpl.to_unsafe,
      c_messages.to_unsafe,
      messages.size,
      add_assistant,
      buffer.to_unsafe,
      required_size
    )

    raise TemplateError.new("chat template is not recognized by llama.cpp") if written < 0
    raise TemplateError.new("chat template output exceeded allocated buffer") if written > required_size

    String.new(buffer.to_unsafe, written)
  end

  # Gets the list of built-in chat templates
  #
  # Returns:
  # - Array of template names
  def self.builtin_chat_templates : Array(String)
    capacity = 100
    output = Pointer(LibC::Char*).malloc(capacity)
    count = LibLlama.llama_chat_builtin_templates(output, capacity)
    raise TemplateError.new("failed to enumerate built-in chat templates") if count < 0
    if count > 10_000
      raise TemplateError.new("built-in chat template count exceeds the safety limit")
    end

    if count > capacity
      capacity = count
      output = Pointer(LibC::Char*).malloc(capacity)
      count = LibLlama.llama_chat_builtin_templates(output, capacity)
      raise TemplateError.new("failed to enumerate built-in chat templates") if count < 0
      if count > capacity
        raise TemplateError.new("built-in chat template count exceeded the allocated buffer")
      end
    end

    result = [] of String
    count.times do |i|
      result << String.new(output[i])
    end

    result
  end

  # Transactional conversation history over a reusable Session.
  class Chat
    include NativeResource

    def initialize(model : Model, system : String? = nil, template : String? = nil, context_options : ContextOptions = ContextOptions.new)
      @model = model
      @mutex = Mutex.new
      @running = false
      @session = model.session(context_options)
      @template = template || model.chat_template
      unless @template
        @session.close
        raise TemplateError.new("model has no chat template and none was provided")
      end
      @history = [] of Message
      @history << Message.system(system) if system
    end

    def ask(content : String, options : GenerationOptions = GenerationOptions.new, commit_partial : Bool = false, &block : GenerationChunk ->) : Generation
      operation_started = false
      candidate = begin_turn!(content)
      operation_started = true
      prompt = Llama.apply_chat_template(@template, candidate.map(&.to_chat_message), true)
      @session.reset
      result = @session.generate(prompt, options, &block)

      if result.finish_reason.cancelled? && !commit_partial
        @session.reset
      else
        @mutex.synchronize do
          @history = candidate + [Message.assistant(result.text)]
        end
      end
      result
    rescue ex
      # A rejected concurrent turn never acquired this Chat transaction. In
      # that case the active turn still owns the Session, so attempting a
      # rollback here would both interfere with it and mask Chat's BusyError
      # with Session's BusyError.
      @session.reset if operation_started && !closed?
      raise ex
    ensure
      end_turn! if operation_started
    end

    def ask(content : String, options : GenerationOptions = GenerationOptions.new, commit_partial : Bool = false) : Generation
      ask(content, options, commit_partial) { |_chunk| }
    end

    def history : Array(Message)
      @mutex.synchronize do
        ensure_open!
        @history.dup
      end
    end

    # Clears all history, including the system message supplied at construction.
    def clear : Nil
      @mutex.synchronize do
        ensure_open!
        raise BusyError.new(self.class.to_s) if @running
        @session.reset
        @history.clear
      end
    end

    def close : Nil
      @mutex.synchronize do
        return if closed?
        raise BusyError.new(self.class.to_s) if @running
        @session.close
      end
    end

    def free : Nil
      close
    end

    def closed? : Bool
      @session.closed?
    end

    def to_unsafe
      @session.to_unsafe
    end

    def finalize
      close
    rescue
    end

    private def begin_turn!(content : String) : Array(Message)
      acquired = false
      candidate = @mutex.synchronize do
        ensure_open!
        raise BusyError.new(self.class.to_s) if @running
        @running = true
        acquired = true
        @history + [Message.user(content)]
      end
      @model.begin_operation!
      candidate
    rescue ex
      @mutex.synchronize { @running = false } if acquired
      raise ex
    end

    private def end_turn! : Nil
      @model.end_operation!
      @mutex.synchronize { @running = false }
    end
  end
end
