# Crystal bindings for llama.cpp
#
# This module provides Crystal bindings for the llama.cpp library,
# allowing you to use LLaMA models in your Crystal applications.

require "./llama/lib_llama"
require "./llama/vocab"
require "./llama/model"
require "./llama/context"

module Llama
  VERSION = "0.1.0"

  # Returns the llama.cpp system information
  def self.system_info : String
    String.new(LibLlama.llama_print_system_info)
  end

  # A simple example of text generation
  #
  # Parameters:
  # - model_path: Path to the model file (.gguf format)
  # - prompt: The input prompt
  # - max_tokens: Maximum number of tokens to generate
  # - temperature: Sampling temperature (0.0 = greedy, 1.0 = more random)
  #
  # Returns:
  # - The generated text
  def self.generate(model_path : String, prompt : String, max_tokens : Int32 = 128, temperature : Float32 = 0.8) : String
    model = Model.new(model_path)
    context = model.context
    context.generate(prompt, max_tokens, temperature)
  end
end
