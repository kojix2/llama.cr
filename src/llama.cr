# Llama - Crystal bindings for llama.cpp
#
# This module provides high-level and low-level APIs for working with LLMs via llama.cpp.
#
# Features:
# - Low-level bindings to the llama.cpp C API
# - High-level Crystal wrapper classes for easy usage
# - Memory management for C resources
# - Simple text generation interface
# - Advanced sampling methods (Min-P, Typical, Mirostat, etc.)
# - Batch processing for efficient token handling
# - KV cache management for optimized inference
# - State saving and loading
#
# Backend Initialization:
# - Llama.init: Thread-safe, idempotent initialization of the llama.cpp backend.
#   You do not need to call this manually in most cases; it is called automatically
#   when you create a Model or Context. However, you may call it explicitly if you
#   want to initialize the backend before any model/context is created.
#
# - Llama.uninit: Thread-safe, idempotent finalization of the llama.cpp backend.
#   Call this if you want to explicitly release all backend resources before program exit.
#
# Example:
#   Llama.init         # (optional, usually not needed)
#   model = Llama::Model.new("model.gguf")
#   context = Llama::Context.new(model)
#   # ... use model/context ...
#   Llama.uninit       # (optional, usually not needed)
#
# The backend is always initialized only once, even if called from multiple threads.
# All resources are released only once, even if uninit is called multiple times.
#
# Basic Usage:
#   require "llama"
#
#   # Load a model
#   model = Llama::Model.new("/path/to/model.gguf")
#
#   # Create a context
#   context = model.context
#
#   # Generate text
#   response = context.generate("Once upon a time", max_tokens: 100, temperature: 0.8)
#   puts response
#
#   # Or use the convenience method
#   response = Llama.generate("/path/to/model.gguf", "Once upon a time")
#   puts response
#
# Advanced Sampling:
#   chain = Llama::SamplerChain.new
#   chain.add(Llama::Sampler::TopK.new(40))
#   chain.add(Llama::Sampler::MinP.new(0.05, 1))
#   chain.add(Llama::Sampler::Temp.new(0.8))
#   chain.add(Llama::Sampler::Dist.new(42))
#
#   result = context.generate_with_sampler("Write a poem:", chain, 150)
#
# System Info:
#   info = Llama.system_info
#   puts info
#
# Tokenization Utility:
#   model = Llama::Model.new("/path/to/model.gguf")
#   result = Llama.tokenize_and_format(model.vocab, "Hello, world!", ids_only: true)
#   puts result # Prints "[1, 2, 3, ...]"

require "./llama/lib_llama"
require "./llama/error"
require "./llama/vocab"
require "./llama/model"
require "./llama/kv_cache"
require "./llama/batch"
require "./llama/state"
require "./llama/adapter_lora"
require "./llama/context"
require "./llama/sampler"

module Llama
  VERSION                      = {{ `shards version #{__DIR__}`.chomp.stringify }}
  LLAMA_CPP_COMPATIBLE_VERSION = read_file("#{__DIR__}/LLAMA_VERSION").chomp

  # ==== Native constants (wrapped for user convenience) ====
  DEFAULT_SEED    = LibLlama::LLAMA_DEFAULT_SEED
  TOKEN_NULL      = LibLlama::LLAMA_TOKEN_NULL
  FILE_MAGIC_GGLA = LibLlama::LLAMA_FILE_MAGIC_GGLA
  FILE_MAGIC_GGSN = LibLlama::LLAMA_FILE_MAGIC_GGSN
  FILE_MAGIC_GGSQ = LibLlama::LLAMA_FILE_MAGIC_GGSQ
  SESSION_MAGIC   = LibLlama::LLAMA_SESSION_MAGIC
  SESSION_VERSION = LibLlama::LLAMA_SESSION_VERSION

  # Mutex for backend initialization/finalization (required for type inference)
  @@backend_mutex : Mutex = Mutex.new
  @@backend_initialized = false

  # Log level constants (from llama.cpp / ggml)
  LOG_LEVEL_DEBUG   = 0
  LOG_LEVEL_INFO    = 1
  LOG_LEVEL_WARNING = 2
  LOG_LEVEL_ERROR   = 3
  LOG_LEVEL_NONE    = 4 # No logging

  # Internal variables
  @@log_level = LOG_LEVEL_INFO # Default is INFO
  @@log_box : Pointer(Void)? = nil

  # Set the log level
  #
  # Parameters:
  # - level : Int32 - log level (0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR, 4=NONE)
  #
  # Example:
  #   Llama.log_level = Llama::LOG_LEVEL_ERROR  # Only show errors
  #   Llama.log_level = Llama::LOG_LEVEL_NONE   # Disable all logging
  def self.log_level=(level : Int32)
    @@log_level = level
    setup_default_logger
  end

  # Get the current log level
  #
  # Returns:
  # - The current log level
  def self.log_level
    @@log_level
  end

  # Internal method: Set up the default logger
  private def self.setup_default_logger
    log_set do |level, message|
      STDERR.print message if level >= @@log_level && level < LOG_LEVEL_NONE
    end
  end

  # Set a custom log callback
  #
  # The block receives:
  # - level : Int32 - log level (0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR)
  # - message : String - log message
  #
  # Example:
  #   Llama.log_set do |level, message|
  #     if level >= Llama::LOG_LEVEL_ERROR
  #       STDERR.print message
  #     end
  #   end
  def self.log_set(&block : Int32, String ->)
    boxed = Box.box(block)
    @@log_box = boxed

    LibLlama.llama_log_set(
      ->(level : Int32, text : LibC::Char*, user_data : Void*) {
        user_callback = Box(Proc(Int32, String, Nil)).unbox(user_data)
        msg = String.new(text)
        user_callback.call(level, msg)
        nil
      },
      boxed
    )
  end

  # Returns the llama.cpp system information
  #
  # This method provides information about the llama.cpp build,
  # including BLAS configuration, CPU features, and GPU support.
  #
  # ```
  # info = Llama.system_info
  # puts info
  # ```
  #
  # Returns:
  # - A string containing system information
  def self.system_info : String
    String.new(LibLlama.llama_print_system_info)
  end

  # Process escape sequences in a string
  #
  # This method processes common escape sequences like \n, \t, etc.
  # in a string, converting them to their actual character representations.
  #
  # ```
  # text = Llama.process_escapes("Hello\\nWorld")
  # puts text # Prints "Hello" and "World" on separate lines
  # ```
  #
  # Parameters:
  # - text: The input string containing escape sequences
  #
  # Returns:
  # - A new string with escape sequences processed
  def self.process_escapes(text : String) : String
    text.gsub(/\\([nrt\\"])/) do |match|
      case match[1]
      when 'n'  then "\n"
      when 'r'  then "\r"
      when 't'  then "\t"
      when '\\' then "\\"
      when '"'  then "\""
      else           match
      end
    end
  end

  # Tokenize text and return formatted output
  #
  # This is a convenience method that tokenizes text and returns
  # a formatted string representation of the tokens.
  #
  # ```
  # model = Llama::Model.new("/path/to/model.gguf")
  # result = Llama.tokenize_and_format(model.vocab, "Hello, world!", ids_only: true)
  # puts result # Prints "[1, 2, 3, ...]"
  # ```
  #
  # Parameters:
  # - vocab: The vocabulary to use for tokenization
  # - text: The text to tokenize
  # - add_bos: Whether to add BOS token (default: true)
  # - parse_special: Whether to parse special tokens (default: true)
  # - ids_only: Whether to return only token IDs (default: false)
  #
  # Returns:
  # - A formatted string representation of the tokens
  def self.tokenize_and_format(
    vocab : Vocab,
    text : String,
    add_bos : Bool = true,
    parse_special : Bool = true,
    ids_only : Bool = false,
  ) : String
    tokens = vocab.tokenize(text, add_bos, parse_special)

    if ids_only
      "[" + tokens.map(&.to_s).join(", ") + "]"
    else
      tokens.map { |t| vocab.format_token(t) }.join("\n")
    end
  end

  # Generates text from a prompt using a model
  #
  # This is a convenience method that loads a model, creates a context,
  # and generates text in a single call.
  #
  # ```
  # response = Llama.generate(
  #   "/path/to/model.gguf",
  #   "Once upon a time",
  #   max_tokens: 100,
  #   temperature: 0.7
  # )
  # puts response
  # ```
  #
  # Parameters:
  # - model_path: Path to the model file (.gguf format)
  # - prompt: The input prompt
  # - max_tokens: Maximum number of tokens to generate (must be positive)
  # - temperature: Sampling temperature (0.0 = greedy, 1.0 = more random)
  #
  # Returns:
  # - The generated text
  #
  # Raises:
  # - ArgumentError if parameters are invalid
  # - Llama::Model::Error if model loading fails
  # - Llama::Context::Error if text generation fails
  def self.generate(model_path : String, prompt : String, max_tokens : Int32 = 128, temperature : Float32 = 0.8) : String
    # Validate parameters
    raise ArgumentError.new("max_tokens must be positive") if max_tokens <= 0
    raise ArgumentError.new("temperature must be non-negative") if temperature < 0

    model = Model.new(model_path)
    context = model.context
    context.generate(prompt, max_tokens, temperature)
  end

  # Thread-safe, idempotent initialization of the llama.cpp backend.
  # You do not need to call this manually in most cases.
  def self.init
    @@backend_mutex.synchronize do
      unless @@backend_initialized
        LibLlama.llama_backend_init
        @@backend_initialized = true
      end
    end
  end

  # Thread-safe, idempotent finalization of the llama.cpp backend.
  # Call this if you want to explicitly release all backend resources before program exit.
  def self.uninit
    @@backend_mutex.synchronize do
      if @@backend_initialized
        LibLlama.llama_backend_free
        @@backend_initialized = false
      end
    end
  end

  # Returns the current time in microseconds since the Unix epoch (llama.cpp compatible).
  #
  # This is a high-level wrapper for LibLlama.llama_time_us.
  #
  # ```
  # t0 = Llama.time_us
  # # ... some processing ...
  # t1 = Llama.time_us
  # elapsed_ms = (t1 - t0) / 1000.0
  # puts "Elapsed: #{elapsed_ms} ms"
  # ```
  #
  # Returns:
  # - Int64: microseconds since epoch
  def self.time_us : Int64
    LibLlama.llama_time_us
  end

  # Returns the current time in milliseconds since the Unix epoch (llama.cpp compatible).
  #
  # ```
  # t0 = Llama.time_ms
  # # ... some processing ...
  # t1 = Llama.time_ms
  # elapsed = t1 - t0
  # puts "Elapsed: #{elapsed} ms"
  # ```
  #
  # Returns:
  # - Int64: milliseconds since epoch
  def self.time_ms : Int64
    LibLlama.llama_time_us // 1000
  end

  # Measures elapsed time in milliseconds for a block using llama.cpp's clock.
  #
  # ```
  # elapsed = Llama.measure_ms do
  #   # ... code to measure ...
  # end
  # puts "Elapsed: #{elapsed} ms"
  # ```
  #
  # Returns:
  # - Float64: elapsed milliseconds
  def self.measure_ms(&)
    t0 = time_us
    yield
    t1 = time_us
    (t1 - t0) / 1000.0
  end
end
