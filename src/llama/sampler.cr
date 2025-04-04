module Llama
  # Wrapper for the llama_sampler structure
  #
  # The Sampler class is the base class for all sampling methods.
  # Sampling is the process of selecting the next token during text generation.
  # Different sampling methods produce different text characteristics.
  class Sampler
    # Creates a new Sampler instance from a raw pointer
    #
    # Note: This constructor is intended for internal use.
    def initialize(@handle : LibLlama::LlamaSampler*)
    end

    # Returns the raw pointer to the underlying llama_sampler structure
    def to_unsafe
      @handle
    end

    # Frees the resources associated with this sampler
    def finalize
      if @handle && !@handle.null?
        LibLlama.llama_sampler_free(@handle)
        @handle = Pointer(LibLlama::LlamaSampler).null
      end
    end

    @handle : LibLlama::LlamaSampler*
  end

  # Wrapper for a chain of samplers
  #
  # The SamplerChain class allows you to combine multiple sampling methods
  # in a sequence. Each sampler in the chain filters the token probabilities
  # before passing them to the next sampler.
  #
  # Example:
  # ```
  # chain = Llama::SamplerChain.new
  # chain.add(Llama::TopKSampler.new(40))      # First filter with top-k
  # chain.add(Llama::TopPSampler.new(0.95, 1)) # Then apply top-p
  # chain.add(Llama::TempSampler.new(0.8))     # Apply temperature
  # chain.add(Llama::DistSampler.new(42))      # Final distribution sampling
  #
  # # Generate text with the custom sampler chain
  # result = context.generate_with_sampler("Your prompt", chain, 150)
  # ```
  class SamplerChain < Sampler
    # Creates a new SamplerChain with optional parameters
    #
    # Parameters:
    # - params: Optional sampler chain parameters
    #
    # Raises:
    # - Llama::Error if the sampler chain cannot be created
    def initialize(params = nil)
      params ||= LibLlama.llama_sampler_chain_default_params
      @handle = LibLlama.llama_sampler_chain_init(params)
      raise Error.new("Failed to create sampler chain") if @handle.null?
      @samplers = [] of Sampler
    end

    # Adds a sampler to the chain
    #
    # Parameters:
    # - sampler: The sampler to add to the chain
    def add(sampler : Sampler)
      LibLlama.llama_sampler_chain_add(@handle, sampler.to_unsafe)
      @samplers << sampler # Keep reference to prevent GC
    end

    # Samples a token using the sampler chain
    #
    # Parameters:
    # - ctx: The context to sample from
    # - idx: The index of the logits to sample from (-1 for the last token)
    #
    # Returns:
    # - The sampled token
    def sample(ctx : Context, idx : Int32 = -1) : Int32
      LibLlama.llama_sampler_sample(@handle, ctx.to_unsafe, idx)
    end

    # Accepts a token, updating the internal state of the samplers
    #
    # Parameters:
    # - token: The token to accept
    def accept(token : Int32)
      LibLlama.llama_sampler_accept(@handle, token)
    end

    # Frees the resources associated with this sampler chain
    # Overrides the parent class's finalize method to also clean up the samplers array
    def finalize
      # Clear the samplers array to allow GC to collect the samplers
      @samplers.clear if @samplers
      super
    end

    @samplers : Array(Sampler)
  end

  # Top-K sampler
  #
  # The Top-K sampler keeps only the K most likely tokens and zeros out the rest.
  # This is a simple but effective way to filter out unlikely tokens.
  #
  # Example:
  # ```
  # sampler = Llama::TopKSampler.new(40) # Keep only the top 40 tokens
  # ```
  class TopKSampler < Sampler
    # Creates a new Top-K sampler
    #
    # Parameters:
    # - k: The number of top tokens to consider
    #
    # Raises:
    # - Llama::Error if the sampler cannot be created
    def initialize(k : Int32)
      @handle = LibLlama.llama_sampler_init_top_k(k)
      raise Error.new("Failed to create Top-K sampler") if @handle.null?
    end
  end

  # Top-P (nucleus) sampler
  #
  # The Top-P sampler (also known as nucleus sampling) keeps tokens whose
  # cumulative probability exceeds the threshold P. This helps to filter out
  # the long tail of low-probability tokens.
  #
  # Example:
  # ```
  # sampler = Llama::TopPSampler.new(0.95, 1) # Keep tokens until 95% probability is covered
  # ```
  class TopPSampler < Sampler
    # Creates a new Top-P sampler
    #
    # Parameters:
    # - p: The cumulative probability threshold (0.0 to 1.0)
    # - min_keep: Minimum number of tokens to keep
    #
    # Raises:
    # - Llama::Error if the sampler cannot be created
    def initialize(p : Float32, min_keep : Int32 = 1)
      @handle = LibLlama.llama_sampler_init_top_p(p, min_keep)
      raise Error.new("Failed to create Top-P sampler") if @handle.null?
    end
  end

  # Temperature sampler
  #
  # The Temperature sampler adjusts the logits by dividing them by the temperature
  # value. Higher temperatures (>1.0) make the distribution more uniform, leading
  # to more random outputs. Lower temperatures (<1.0) make the distribution more
  # peaked, leading to more deterministic outputs.
  #
  # Example:
  # ```
  # sampler = Llama::TempSampler.new(0.8) # Slightly more deterministic than default
  # ```
  class TempSampler < Sampler
    # Creates a new temperature sampler
    #
    # Parameters:
    # - temp: The temperature value (0.0 = greedy, 1.0 = normal, >1.0 = more random)
    #
    # Raises:
    # - Llama::Error if the sampler cannot be created
    def initialize(temp : Float32)
      @handle = LibLlama.llama_sampler_init_temp(temp)
      raise Error.new("Failed to create temperature sampler") if @handle.null?
    end
  end

  # Distribution sampler (final sampler in a chain)
  #
  # The Distribution sampler is typically the final sampler in a chain.
  # It samples a token from the probability distribution after all other
  # samplers have filtered the logits. This sampler is required to actually
  # select a token from the distribution.
  #
  # Example:
  # ```
  # sampler = Llama::DistSampler.new(42) # Use seed 42 for reproducibility
  # ```
  class DistSampler < Sampler
    # Creates a new distribution sampler
    #
    # Parameters:
    # - seed: Random seed for sampling (default: LLAMA_DEFAULT_SEED)
    #
    # Raises:
    # - Llama::Error if the sampler cannot be created
    def initialize(seed : UInt32 = LibLlama::LLAMA_DEFAULT_SEED)
      @handle = LibLlama.llama_sampler_init_dist(seed)
      raise Error.new("Failed to create distribution sampler") if @handle.null?
    end
  end

  # ===== ADVANCED SAMPLERS =====

  # Min-P sampler
  #
  # The Min-P sampler keeps tokens with probability >= P * max_probability.
  # This is similar to Top-P but uses a minimum probability threshold relative
  # to the most likely token, rather than a cumulative probability threshold.
  #
  # Example:
  # ```
  # sampler = Llama::MinPSampler.new(0.05, 1) # Keep tokens with prob >= 5% of max prob
  # ```
  class MinPSampler < Sampler
    # Creates a new Min-P sampler
    #
    # Parameters:
    # - p: The minimum probability threshold (0.0 to 1.0)
    # - min_keep: Minimum number of tokens to keep
    #
    # Raises:
    # - Llama::Error if the sampler cannot be created
    def initialize(p : Float32, min_keep : Int32 = 1)
      @handle = LibLlama.llama_sampler_init_min_p(p, min_keep)
      raise Error.new("Failed to create Min-P sampler") if @handle.null?
    end
  end

  # Typical sampler
  #
  # The Typical sampler selects tokens based on their "typicality" (entropy).
  # It filters out tokens that are either too predictable or too surprising,
  # leading to more natural and diverse text generation.
  #
  # Based on the paper: https://arxiv.org/abs/2202.00666
  #
  # Example:
  # ```
  # sampler = Llama::TypicalSampler.new(0.95, 1) # Keep tokens with typicality >= 0.95
  # ```
  class TypicalSampler < Sampler
    # Creates a new Typical sampler
    #
    # Parameters:
    # - p: The typicality threshold (0.0 to 1.0)
    # - min_keep: Minimum number of tokens to keep
    #
    # Raises:
    # - Llama::Error if the sampler cannot be created
    def initialize(p : Float32, min_keep : Int32 = 1)
      @handle = LibLlama.llama_sampler_init_typical(p, min_keep)
      raise Error.new("Failed to create Typical sampler") if @handle.null?
    end
  end

  # Mirostat sampler (version 1)
  #
  # The Mirostat sampler dynamically adjusts the temperature to maintain
  # a target entropy level in the generated text. This helps to produce
  # consistent quality output regardless of the context.
  #
  # Based on the paper: https://arxiv.org/abs/2007.14966
  #
  # Example:
  # ```
  # # Create a Mirostat sampler with vocabulary size, seed, target entropy,
  # # learning rate, and context size
  # sampler = Llama::MirostatSampler.new(32000, 42, 5.0, 0.1, 100)
  # ```
  class MirostatSampler < Sampler
    # Creates a new Mirostat sampler
    #
    # Parameters:
    # - n_vocab: Vocabulary size
    # - seed: Random seed
    # - tau: Target entropy (5.0 - 8.0 is a good range)
    # - eta: Learning rate (0.1 is a good default)
    # - m: Number of tokens for estimating entropy (100 is a good default)
    #
    # Raises:
    # - Llama::Error if the sampler cannot be created
    def initialize(n_vocab : Int32, seed : UInt32, tau : Float32, eta : Float32, m : Int32)
      @handle = LibLlama.llama_sampler_init_mirostat(n_vocab, seed, tau, eta, m)
      raise Error.new("Failed to create Mirostat sampler") if @handle.null?
    end
  end

  # Mirostat sampler (version 2)
  #
  # The Mirostat V2 sampler is an improved version of the Mirostat algorithm
  # that requires fewer parameters and is more efficient. It dynamically
  # adjusts sampling to maintain a target entropy level.
  #
  # Based on the paper: https://arxiv.org/abs/2007.14966
  #
  # Example:
  # ```
  # sampler = Llama::MirostatV2Sampler.new(42, 5.0, 0.1)
  # ```
  class MirostatV2Sampler < Sampler
    # Creates a new Mirostat V2 sampler
    #
    # Parameters:
    # - seed: Random seed
    # - tau: Target entropy (5.0 is a good default)
    # - eta: Learning rate (0.1 is a good default)
    #
    # Raises:
    # - Llama::Error if the sampler cannot be created
    def initialize(seed : UInt32, tau : Float32, eta : Float32)
      @handle = LibLlama.llama_sampler_init_mirostat_v2(seed, tau, eta)
      raise Error.new("Failed to create Mirostat V2 sampler") if @handle.null?
    end
  end

  # Grammar sampler
  #
  # The Grammar sampler constrains token generation to follow a formal grammar
  # defined in GBNF format. This is useful for generating structured text like
  # JSON, XML, or code that must follow specific syntax rules.
  #
  # Example:
  # ```
  # # Define a simple grammar for a numbered list
  # grammar = %q{
  #   root ::= list
  #   list ::= item+
  #   item ::= number " " text "\n"
  #   number ::= "1" | "2" | "3" | "4" | "5"
  #   text ::= [a-zA-Z ]+
  # }
  #
  # sampler = Llama::GrammarSampler.new(model.vocab, grammar, "root")
  # ```
  class GrammarSampler < Sampler
    # Creates a new Grammar sampler
    #
    # Parameters:
    # - vocab: The vocabulary to use
    # - grammar_str: The grammar definition string in GBNF format
    # - grammar_root: The root symbol of the grammar
    #
    # Raises:
    # - Llama::Error if the sampler cannot be created
    def initialize(vocab : Vocab, grammar_str : String, grammar_root : String)
      @handle = LibLlama.llama_sampler_init_grammar(
        vocab.to_unsafe,
        grammar_str,
        grammar_root
      )
      raise Error.new("Failed to create Grammar sampler") if @handle.null?

      # Store references to prevent GC
      @vocab = vocab
      @grammar_str = grammar_str
      @grammar_root = grammar_root
    end

    # Overrides the parent class's finalize method to ensure proper cleanup
    def finalize
      # First nullify our references to prevent circular references
      @vocab = nil
      @grammar_str = nil
      @grammar_root = nil

      # Then call the parent's finalize method
      super
    end

    # Instance variables to keep references to prevent GC
    @vocab : Vocab?
    @grammar_str : String?
    @grammar_root : String?
  end

  # Penalties sampler
  #
  # The Penalties sampler applies various penalties to token probabilities
  # to reduce repetition and improve diversity in the generated text.
  # It can penalize recently used tokens, frequent tokens, and more.
  #
  # Example:
  # ```
  # # Apply penalties to the last 64 tokens with a repetition penalty of 1.1
  # sampler = Llama::PenaltiesSampler.new(64, 1.1, 0.0, 0.0)
  # ```
  class PenaltiesSampler < Sampler
    # Creates a new Penalties sampler
    #
    # Parameters:
    # - penalty_last_n: Last n tokens to penalize (0 = disable, -1 = context size)
    # - penalty_repeat: Repetition penalty (1.0 = disabled)
    # - penalty_freq: Frequency penalty (0.0 = disabled)
    # - penalty_present: Presence penalty (0.0 = disabled)
    #
    # Raises:
    # - Llama::Error if the sampler cannot be created
    def initialize(penalty_last_n : Int32, penalty_repeat : Float32, penalty_freq : Float32, penalty_present : Float32)
      @handle = LibLlama.llama_sampler_init_penalties(
        penalty_last_n,
        penalty_repeat,
        penalty_freq,
        penalty_present
      )
      raise Error.new("Failed to create Penalties sampler") if @handle.null?
    end
  end

  # Extended Temperature sampler
  #
  # The Extended Temperature sampler provides more control over temperature
  # sampling with additional parameters for dynamic temperature adjustment.
  # This is based on the paper "Dynamic Temperature for Language Models" (https://arxiv.org/abs/2309.02772).
  #
  # Example:
  # ```
  # # Create a temperature sampler with base temp 0.8, delta 0.5, and exponent 1.0
  # sampler = Llama::TempExtSampler.new(0.8, 0.5, 1.0)
  # ```
  class TempExtSampler < Sampler
    # Creates a new Extended Temperature sampler
    #
    # Parameters:
    # - t: Base temperature value
    # - delta: Temperature delta for dynamic adjustment
    # - exponent: Exponent for the temperature formula
    #
    # Raises:
    # - Llama::Error if the sampler cannot be created
    def initialize(t : Float32, delta : Float32, exponent : Float32)
      @handle = LibLlama.llama_sampler_init_temp_ext(t, delta, exponent)
      raise Error.new("Failed to create Extended Temperature sampler") if @handle.null?
    end
  end

  # Top-N Sigma sampler
  #
  # The Top-N Sigma sampler selects tokens based on their distance from the mean
  # in terms of standard deviations. This is based on the paper "Top-nσ: Not All Logits Are You Need"
  # (https://arxiv.org/pdf/2411.07641).
  #
  # Example:
  # ```
  # sampler = Llama::TopNSigmaSampler.new(2.0) # Keep tokens within 2 standard deviations
  # ```
  class TopNSigmaSampler < Sampler
    # Creates a new Top-N Sigma sampler
    #
    # Parameters:
    # - n: Number of standard deviations to keep
    #
    # Raises:
    # - Llama::Error if the sampler cannot be created
    def initialize(n : Float32)
      @handle = LibLlama.llama_sampler_init_top_n_sigma(n)
      raise Error.new("Failed to create Top-N Sigma sampler") if @handle.null?
    end
  end

  # XTC sampler
  #
  # The XTC sampler combines aspects of several sampling methods for improved
  # text generation quality. It was introduced in the Text Generation WebUI project.
  #
  # Example:
  # ```
  # sampler = Llama::XtcSampler.new(0.3, 0.8, 1, 42)
  # ```
  class XtcSampler < Sampler
    # Creates a new XTC sampler
    #
    # Parameters:
    # - p: Probability threshold
    # - t: Temperature value
    # - min_keep: Minimum number of tokens to keep
    # - seed: Random seed for sampling
    #
    # Raises:
    # - Llama::Error if the sampler cannot be created
    def initialize(p : Float32, t : Float32, min_keep : Int32, seed : UInt32 = LibLlama::LLAMA_DEFAULT_SEED)
      @handle = LibLlama.llama_sampler_init_xtc(p, t, min_keep, seed)
      raise Error.new("Failed to create XTC sampler") if @handle.null?
    end
  end

  # Infill sampler
  #
  # The Infill sampler is designed for fill-in-the-middle (FIM) tasks.
  # It helps to generate text that fits naturally between existing content.
  #
  # Example:
  # ```
  # sampler = Llama::InfillSampler.new(model.vocab)
  # ```
  class InfillSampler < Sampler
    # Creates a new Infill sampler
    #
    # Parameters:
    # - vocab: The vocabulary to use
    #
    # Raises:
    # - Llama::Error if the sampler cannot be created
    def initialize(vocab : Vocab)
      @handle = LibLlama.llama_sampler_init_infill(vocab.to_unsafe)
      raise Error.new("Failed to create Infill sampler") if @handle.null?

      # Store reference to prevent GC
      @vocab = vocab
    end

    # Overrides the parent class's finalize method to ensure proper cleanup
    def finalize
      # First nullify our reference to prevent circular references
      @vocab = nil

      # Then call the parent's finalize method
      super
    end

    # Instance variable to keep reference to prevent GC
    @vocab : Vocab?
  end

  # Grammar Lazy Patterns sampler
  #
  # The Grammar Lazy Patterns sampler is an extension of the Grammar sampler
  # that only applies grammar constraints when triggered by specific patterns
  # or tokens. This is useful for mixed-format generation where grammar
  # constraints should only apply to certain parts of the output.
  #
  # Example:
  # ```
  # # Define a JSON grammar that only activates when the text contains "JSON:"
  # grammar = %q{
  #   root ::= object
  #   object ::= "{" ws (string ":" ws value ("," ws string ":" ws value)*)? "}" ws
  #   array ::= "[" ws (value ("," ws value)*)? "]" ws
  #   value ::= object | array | string | number | "true" | "false" | "null"
  #   string ::= "\"" ([^"\\] | "\\" .)* "\""
  #   number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)? ([eE] [-+]? [0-9]+)?
  #   ws ::= [ \t\n]*
  # }
  #
  # trigger_patterns = ["JSON:"]
  # sampler = Llama::GrammarLazyPatternsSampler.new(
  #   model.vocab, grammar, "root", trigger_patterns
  # )
  # ```
  class GrammarLazyPatternsSampler < Sampler
    # Creates a new Grammar Lazy Patterns sampler
    #
    # Parameters:
    # - vocab: The vocabulary to use
    # - grammar_str: The grammar definition string in GBNF format
    # - grammar_root: The root symbol of the grammar
    # - trigger_patterns: Array of string patterns that will trigger the grammar
    # - trigger_tokens: Array of token IDs that will trigger the grammar
    #
    # Raises:
    # - Llama::Error if the sampler cannot be created
    def initialize(
      vocab : Vocab,
      grammar_str : String,
      grammar_root : String,
      trigger_patterns : Array(String) = [] of String,
      trigger_tokens : Array(Int32) = [] of Int32,
    )
      # Convert trigger patterns to C-style array
      patterns_ptr = Pointer(LibC::Char*).malloc(trigger_patterns.size + 1)
      trigger_patterns.each_with_index do |pattern, i|
        patterns_ptr[i] = pattern.to_unsafe
      end
      patterns_ptr[trigger_patterns.size] = Pointer(LibC::Char).null

      # Convert trigger tokens to C-style array
      tokens_ptr = Pointer(LibLlama::LlamaToken).null
      if trigger_tokens.size > 0
        tokens_ptr = Pointer(LibLlama::LlamaToken).malloc(trigger_tokens.size)
        trigger_tokens.each_with_index do |token, i|
          tokens_ptr[i] = token
        end
      end

      @handle = LibLlama.llama_sampler_init_grammar_lazy_patterns(
        vocab.to_unsafe,
        grammar_str,
        grammar_root,
        patterns_ptr,
        trigger_patterns.size,
        tokens_ptr,
        trigger_tokens.size
      )
      raise Error.new("Failed to create Grammar Lazy Patterns sampler") if @handle.null?

      # Store references to prevent GC
      @vocab = vocab
      @grammar_str = grammar_str
      @grammar_root = grammar_root
      @trigger_patterns = trigger_patterns
      @trigger_tokens = trigger_tokens
    end

    # Overrides the parent class's finalize method to ensure proper cleanup
    def finalize
      # First nullify our references to prevent circular references
      @vocab = nil
      @grammar_str = nil
      @grammar_root = nil
      @trigger_patterns = nil
      @trigger_tokens = nil

      # Then call the parent's finalize method
      super
    end

    # Instance variables to keep references to prevent GC
    @vocab : Vocab?
    @grammar_str : String?
    @grammar_root : String?
    @trigger_patterns : Array(String)?
    @trigger_tokens : Array(Int32)?
  end
end
