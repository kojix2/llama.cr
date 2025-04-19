# Crystal bindings for llama.cpp
#
# This module provides Crystal bindings for the llama.cpp library,
# allowing you to use LLaMA models in your Crystal applications.
#
# ## Features
#
# - Low-level bindings to the llama.cpp C API
# - High-level Crystal wrapper classes for easy usage
# - Memory management for C resources
# - Simple text generation interface
# - Advanced sampling methods (Min-P, Typical, Mirostat, etc.)
# - Batch processing for efficient token handling
# - KV cache management for optimized inference
# - State saving and loading
#
# ## Basic Usage
#
# ```
# require "llama"
#
# # Load a model
# model = Llama::Model.new("/path/to/model.gguf")
#
# # Create a context
# context = model.context
#
# # Generate text
# response = context.generate("Once upon a time", max_tokens: 100, temperature: 0.8)
# puts response
#
# # Or use the convenience method
# response = Llama.generate("/path/to/model.gguf", "Once upon a time")
# puts response
# ```
#
# ## Advanced Sampling
#
# ```
# chain = Llama::Sampler::Chain.new
# chain.add(Llama::Sampler::TopK.new(40))
# chain.add(Llama::Sampler::MinP.new(0.05, 1))
# chain.add(Llama::Sampler::Temp.new(0.8))
# chain.add(Llama::Sampler::Dist.new(42))
#
# result = context.generate_with_sampler("Write a poem:", chain, 150)
# ```

require "./llama/lib_llama"
require "./llama/error"
require "./llama/vocab"
require "./llama/model"
require "./llama/kv_cache"
require "./llama/batch"
require "./llama/state"
require "./llama/context"
require "./llama/sampler"

module Llama
  VERSION                      = "0.1.0"
  LLAMA_CPP_COMPATIBLE_VERSION = read_file("#{__DIR__}/LLAMA_VERSION").chomp

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
  # - Llama::ModelError if model loading fails
  # - Llama::ContextError if text generation fails
  def self.generate(model_path : String, prompt : String, max_tokens : Int32 = 128, temperature : Float32 = 0.8) : String
    # Validate parameters
    raise ArgumentError.new("max_tokens must be positive") if max_tokens <= 0
    raise ArgumentError.new("temperature must be non-negative") if temperature < 0

    model = Model.new(model_path)
    context = model.context
    context.generate(prompt, max_tokens, temperature)
  end
end
