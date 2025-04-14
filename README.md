# llama.cr

[![test](https://github.com/kojix2/llama.cr/actions/workflows/test.yml/badge.svg)](https://github.com/kojix2/llama.cr/actions/workflows/test.yml)
[![docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://kojix2.github.io/llama.cr)
[![Lines of Code](https://img.shields.io/endpoint?url=https%3A%2F%2Ftokei.kojix2.net%2Fbadge%2Fgithub%2Fkojix2%2Fllama.cr%2Flines)](https://tokei.kojix2.net/github/kojix2/llama.cr)

Crystal bindings for [llama.cpp](https://github.com/ggml-org/llama.cpp), a C/C++ implementation of LLaMA, Falcon, GPT-2, and other large language models.

Please check the [LLAMA_VERSION](LLAMA_VERSION) file for the current compatible version of llama.cpp.

## Features

- Low-level bindings to the llama.cpp C API
- High-level Crystal wrapper classes for easy usage
- Memory management for C resources
- Simple text generation interface
- Advanced sampling methods (Min-P, Typical, Mirostat, etc.)
- Batch processing for efficient token handling
- KV cache management for optimized inference
- State saving and loading

## Installation

### Prerequisites

Before using this shard, you need to have llama.cpp compiled and installed on your system:

1. Clone and build llama.cpp using CMake:

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
mkdir build && cd build
cmake ..
cmake --build . --config Release
```

2. Install the library to your system:

```bash
# On Linux
sudo cmake --install .
sudo ldconfig

# On macOS
sudo cmake --install .
```

3. Alternatively, you can specify the library location without installing:

```bash
# Set the library path at compile time
crystal build examples/text_generation.cr --link-flags="-L/path/to/llama.cpp/build/bin"

# Or set the library path at runtime
LD_LIBRARY_PATH=/path/to/llama.cpp/build/bin ./text_generation /path/to/model.gguf "Your prompt here"
```

### Obtaining GGUF Model Files

You'll need a model file in GGUF format. For testing, smaller quantized models (1-3B parameters) with Q4_K_M quantization are recommended.

Popular options:

- [TinyLlama 1.1B](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF) (good for testing)
- [Llama 3 8B Instruct](https://huggingface.co/TheBloke/Llama-3-8B-Instruct-GGUF)
- [Mistral 7B Instruct v0.2](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF)

### Adding to Your Project

1. Add the dependency to your `shard.yml`:

```yaml
dependencies:
  llama:
    github: kojix2/llama.cr
```

2. Run `shards install`

## Usage

### Basic Text Generation

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

### Advanced Sampling

```crystal
require "llama"

model = Llama::Model.new("/path/to/model.gguf")
context = model.context

# Create a sampler chain with multiple sampling methods
chain = Llama::SamplerChain.new
chain.add(Llama::TopKSampler.new(40))      # First filter with top-k
chain.add(Llama::MinPSampler.new(0.05, 1)) # Then apply min-p
chain.add(Llama::TempSampler.new(0.8))     # Apply temperature
chain.add(Llama::DistSampler.new(42))      # Final distribution sampling

# Generate text with the custom sampler chain
result = context.generate_with_sampler("Write a short poem about AI:", chain, 150)
puts result
```

### Chat Conversations

```crystal
require "llama"
require "llama/chat"

model = Llama::Model.new("/path/to/model.gguf")
context = model.context

# Create a chat conversation
messages = [
  Llama::ChatMessage.new("system", "You are a helpful assistant."),
  Llama::ChatMessage.new("user", "Hello, who are you?")
]

# Generate a response
response = context.chat(messages)
puts "Assistant: #{response}"

# Continue the conversation
messages << Llama::ChatMessage.new("assistant", response)
messages << Llama::ChatMessage.new("user", "Tell me a joke")
response = context.chat(messages)
puts "Assistant: #{response}"
```

### Embeddings

```crystal
require "llama"

model = Llama::Model.new("/path/to/model.gguf")

# Create a context with embeddings enabled
params = Llama::Context.default_params
params.embeddings = true
context = model.context(params)

# Get embeddings for text
text = "Hello, world!"
tokens = model.vocab.tokenize(text)
batch = Llama::Batch.get_one(tokens)
context.decode(batch)
embeddings = context.get_embeddings_seq(0)

puts "Embedding dimension: #{embeddings.size}"
```

## Examples

The `examples` directory contains sample code demonstrating various features:

- `text_generation.cr` - Basic text generation
- `chat.cr` - Chat conversations with models
- `tokenization.cr` - Tokenization and vocabulary features
- `sampling.cr` - Custom token sampling
- `advanced_sampling.cr` - Advanced sampling methods
- `embeddings.cr` - Text embeddings
- `advanced.cr` - KV cache and state management

## API Documentation

### Core Classes

- **Llama::Model** - Represents a loaded LLaMA model
- **Llama::Context** - Handles inference state for a model
- **Llama::Vocab** - Provides access to the model's vocabulary
- **Llama::Batch** - Manages batches of tokens for efficient processing
- **Llama::KvCache** - Controls the key-value cache for optimized inference
- **Llama::State** - Handles saving and loading model state
- **Llama::SamplerChain** - Combines multiple sampling methods

### Samplers

- **Llama::TopKSampler** - Keeps only the top K most likely tokens
- **Llama::TopPSampler** - Nucleus sampling (keeps tokens until cumulative probability exceeds P)
- **Llama::TempSampler** - Applies temperature to logits
- **Llama::DistSampler** - Samples from the final probability distribution
- **Llama::MinPSampler** - Keeps tokens with probability >= P \* max_probability
- **Llama::TypicalSampler** - Selects tokens based on their "typicality" (entropy)
- **Llama::MirostatSampler** - Dynamically adjusts sampling to maintain target entropy
- **Llama::PenaltiesSampler** - Applies penalties to reduce repetition

## Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for development guidelines.

## Contributing

1. Fork it (<https://github.com/kojix2/llama.cr/fork>)
2. Create your feature branch (`git checkout -b my-new-feature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin my-new-feature`)
5. Create a new Pull Request

## License

This project is available under the MIT License. See the LICENSE file for more info.
