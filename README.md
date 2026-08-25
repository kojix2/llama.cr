# llama.cr

[![test](https://github.com/kojix2/llama.cr/actions/workflows/test.yml/badge.svg)](https://github.com/kojix2/llama.cr/actions/workflows/test.yml)
[![examples](https://github.com/kojix2/llama.cr/actions/workflows/examples.yml/badge.svg)](https://github.com/kojix2/llama.cr/actions/workflows/examples.yml)
[![docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://kojix2.github.io/llama.cr)
[![Lines of Code](https://img.shields.io/endpoint?url=https%3A%2F%2Ftokei.kojix2.net%2Fbadge%2Fgithub%2Fkojix2%2Fllama.cr%2Flines)](https://tokei.kojix2.net/github/kojix2/llama.cr)
![Static Badge](https://img.shields.io/badge/PURE-VIBE_CODING-magenta)

Crystal bindings for [llama.cpp](https://github.com/ggml-org/llama.cpp), a C/C++ implementation of LLaMA, Falcon, GPT-2, and other large language models.

The version in `shard.yml` corresponds to the compatible llama.cpp build number.
For example, shard version `0.10566.0` targets llama.cpp build `b10566`, which
is the stable release [v0.2.0](https://github.com/ggml-org/llama.cpp/releases/tag/v0.2.0).

This project is under active development and may change rapidly.

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

Install `llama.cpp` first, then add this shard.

### 1. Install llama.cpp

macOS (Homebrew)

```sh
brew install llama.cpp
export LLAMA_LIB_DIR="$(brew --prefix llama.cpp)/lib"
```

Linux (prebuilt release matching this shard version)

```sh
VERSION="$(shards version)"
BUILD="$(echo "$VERSION" | sed -E 's/^0\.([0-9]+)\.[0-9]+$/\1/')"
LLAMA_BUILD="b${BUILD}"
curl -L "https://github.com/ggml-org/llama.cpp/releases/download/${LLAMA_BUILD}/llama-${LLAMA_BUILD}-bin-ubuntu-x64.tar.gz" -o llama.tar.gz
tar -xzf llama.tar.gz
sudo cp llama-${LLAMA_BUILD}/*.so* /usr/local/lib/
sudo ldconfig
```

### 2. Add to your project

```yaml
dependencies:
  llama:
    github: kojix2/llama.cr
    version: 0.<build>.<patch>
```

Then run:

```sh
shards install
```

Pin an exact version because llama.cpp updates can include breaking changes between build numbers.

### 3. Build and run

Linux:

```sh
export LLAMA_LIB_DIR=/path/to/llama.cpp/lib
LIBRARY_PATH="$LLAMA_LIB_DIR" crystal build examples/simple.cr \
  --link-flags "-L$LLAMA_LIB_DIR -Wl,-rpath,$LLAMA_LIB_DIR -lllama -lggml"
LD_LIBRARY_PATH="$LLAMA_LIB_DIR" ./simple --model models/tiny_model.gguf
```

macOS:

```sh
export LLAMA_LIB_DIR=/path/to/llama.cpp/lib
LIBRARY_PATH="$LLAMA_LIB_DIR" crystal build examples/simple.cr \
  --link-flags "-L$LLAMA_LIB_DIR -Wl,-rpath,$LLAMA_LIB_DIR -lllama -lggml"
DYLD_LIBRARY_PATH="$LLAMA_LIB_DIR" ./simple --model models/tiny_model.gguf
```

If needed, set extra runtime variables:

If backend auto-detection fails in newer llama.cpp builds, set `GGML_BACKEND_PATH` to a backend shared library file (not a directory), for example:

```sh
export GGML_BACKEND_PATH="$LLAMA_LIB_DIR/libggml-cpu-haswell.so"
```

<details>
<summary>Advanced setup</summary>

Build from source:

```sh
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
VERSION="$(shards version ..)"
BUILD="$(echo "$VERSION" | sed -E 's/^0\.([0-9]+)\.[0-9]+$/\1/')"
LLAMA_BUILD="b${BUILD}"
git checkout "${LLAMA_BUILD}"
mkdir build && cd build
cmake .. && cmake --build . --config Release
sudo cmake --install . && sudo ldconfig
```

Example for local development/tests:

```sh
MODEL_PATH=/path/to/model.gguf \
LIBRARY_PATH="$LLAMA_LIB_DIR" \
LD_LIBRARY_PATH="$LLAMA_LIB_DIR" \
GGML_BACKEND_PATH="$LLAMA_LIB_DIR/libggml-cpu-haswell.so" \
crystal spec
```

</details>

### Obtaining GGUF Model Files

You'll need a model file in GGUF format. For testing, smaller quantized models (1-3B parameters) with Q4_K_M quantization are recommended.

Popular options:

- [TinyLlama 1.1B](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF) [[raw]](https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf)
- [Llama 3 8B Instruct](https://huggingface.co/mmnga/Meta-Llama-3-70B-Instruct-gguf)
- [Mistral 7B Instruct v0.2](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF)

## Usage

### Backend Lifetime

`Llama.init` is called automatically when a model or context is created, so most
applications do not need to call it manually.

`Llama.uninit` is optional and usually not needed. It is intended only for
controlled teardown after all `Llama::Model` and `Llama::Context` instances have
been finalized. Calling it while models or contexts are still alive raises an
error, because their finalizers may still need the llama.cpp backend.

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
chain.add(Llama::Sampler::TopK.new(40))
chain.add(Llama::Sampler::MinP.new(0.05, 1))
chain.add(Llama::Sampler::Temp.new(0.8))
chain.add(Llama::Sampler::Dist.new(42))

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
context = model.context(embeddings: true)

# Get embeddings for text
text = "Hello, world!"
tokens = model.vocab.tokenize(text)
batch = Llama::Batch.from_tokens(tokens)
context.decode(batch)
embeddings = context.get_embeddings_seq(0)

puts "Embedding dimension: #{embeddings.size}"
```

### Utilities

#### System Info

```crystal
puts Llama.system_info
```

#### Tokenization Utility

```crystal
model = Llama::Model.new("/path/to/model.gguf")
puts Llama.tokenize_and_format(model.vocab, "Hello, world!", ids_only: true)
```

## Examples

The `examples` directory contains sample code demonstrating various features:

- `simple.cr` - Basic text generation
- `chat.cr` - Chat conversations with models
- `tokenize.cr` - Tokenization and vocabulary features

## API Documentation

See [kojix2.github.io/llama.cr](https://kojix2.github.io/llama.cr) for full API docs.

### Core Classes

- [Llama::Model](https://kojix2.github.io/llama.cr/Llama/Model.html) - Represents a loaded LLaMA model
- [Llama::Context](https://kojix2.github.io/llama.cr/Llama/Context.html) - Handles inference state for a model
- [Llama::Vocab](https://kojix2.github.io/llama.cr/Llama/Vocab.html) - Provides access to the model's vocabulary
- [Llama::Batch](https://kojix2.github.io/llama.cr/Llama/Batch.html) - Manages batches of tokens for efficient processing
- [Llama::Memory](https://kojix2.github.io/llama.cr/Llama/Memory.html) - Controls KV cache memory and related operations
- [Llama::State](https://kojix2.github.io/llama.cr/Llama/State.html) - Handles saving and loading model state
- [Llama::SamplerChain](https://kojix2.github.io/llama.cr/Llama/SamplerChain.html) - Combines multiple sampling methods

### Samplers

- [Llama::Sampler::TopK](https://kojix2.github.io/llama.cr/Llama/Sampler/TopK.html) - Keeps only the top K most likely tokens
- [Llama::Sampler::TopP](https://kojix2.github.io/llama.cr/Llama/Sampler/TopP.html) - Nucleus sampling (keeps tokens until cumulative probability exceeds P)
- [Llama::Sampler::Temp](https://kojix2.github.io/llama.cr/Llama/Sampler/Temp.html) - Applies temperature to logits
- [Llama::Sampler::Dist](https://kojix2.github.io/llama.cr/Llama/Sampler/Dist.html) - Samples from the final probability distribution
- [Llama::Sampler::MinP](https://kojix2.github.io/llama.cr/Llama/Sampler/MinP.html) - Keeps tokens with probability >= P \* max_probability
- [Llama::Sampler::Typical](https://kojix2.github.io/llama.cr/Llama/Sampler/Typical.html) - Selects tokens based on their "typicality" (entropy)
- [Llama::Sampler::Mirostat](https://kojix2.github.io/llama.cr/Llama/Sampler/Mirostat.html) - Dynamically adjusts sampling to maintain target entropy
- [Llama::Sampler::Penalties](https://kojix2.github.io/llama.cr/Llama/Sampler/Penalties.html) - Applies penalties to reduce repetition

## Development

See [DEVELOPMENT.md](DEVELOPMENT.md) for development guidelines.

This software is primarily created through AI-generated code.

Do you need commit rights?

- If you need commit rights to my repository or want to get admin rights and take over the project, please feel free to contact @kojix2.
- Many OSS projects become abandoned because only the founder has commit rights to the original repository.

## Contributing

1. Fork it (<https://github.com/kojix2/llama.cr/fork>)
2. Create your feature branch (`git checkout -b my-new-feature`)
3. Commit your changes (`git commit -am 'Add some feature'`)
4. Push to the branch (`git push origin my-new-feature`)
5. Create a new Pull Request

## License

This project is available under the MIT License. See the LICENSE file for more info.
