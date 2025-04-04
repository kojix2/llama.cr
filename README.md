# llama.cr

Crystal bindings for [llama.cpp](https://github.com/ggml-org/llama.cpp), a C/C++ implementation of LLaMA, Falcon, GPT-2, and other large language models.

## Features

- Low-level bindings to the llama.cpp C API
- High-level Crystal wrapper classes for easy usage
- Memory management for C resources
- Simple text generation interface

## Installation

### Prerequisites

Before using this shard, you need to have llama.cpp compiled and installed on your system:

1. Clone and build llama.cpp:
   ```bash
   git clone https://github.com/ggml-org/llama.cpp.git
   cd llama.cpp
   make
   ```

2. Install the library to your system (optional):
   ```bash
   sudo make install
   ```

### Adding to Your Project

1. Add the dependency to your `shard.yml`:

   ```yaml
   dependencies:
     llama:
       github: kojix2/llama.cr
   ```

2. Run `shards install`

## Usage

```crystal
require "llama"

# Load a model
model = Llama::Model.new("/path/to/model.gguf")

# Create a context
context = model.context

# Generate text
response = context.generate("Once upon a time", max_tokens: 100, temperature: 0.8)
puts response

# Or use the convenience method
response = Llama.generate("/path/to/model.gguf", "Once upon a time")
puts response
```

## API Documentation

### Llama::Model

The `Model` class represents a loaded LLaMA model.

```crystal
# Load a model
model = Llama::Model.new("/path/to/model.gguf")

# Get model information
puts model.n_params  # Number of parameters
puts model.n_embd    # Embedding size
puts model.n_layer   # Number of layers
puts model.n_head    # Number of attention heads
```

### Llama::Context

The `Context` class handles the inference state for a model.

```crystal
# Create a context
context = model.context

# Generate text
response = context.generate("Hello, I am a", max_tokens: 50, temperature: 0.7)
```

### Llama::Vocab

The `Vocab` class provides access to the model's vocabulary.

```crystal
# Get the vocabulary
vocab = model.vocab

# Tokenize text
tokens = vocab.tokenize("Hello, world!")

# Convert tokens back to text
text = tokens.map { |token| vocab.token_to_text(token) }.join
```

## Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for development guidelines.

## Contributing

1. Fork it (<https://github.com/kojix2/llama.cr/fork>)
2. Create your feature branch (`git checkout -b my-new-feature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin my-new-feature`)
5. Create a new Pull Request

## Contributors

- [kojix2](https://github.com/kojix2) - creator and maintainer

## License

This project is available under the MIT License. See the LICENSE file for more info.
