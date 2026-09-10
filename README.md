# llama.cr

[![test](https://github.com/kojix2/llama.cr/actions/workflows/test.yml/badge.svg)](https://github.com/kojix2/llama.cr/actions/workflows/test.yml)
[![examples](https://github.com/kojix2/llama.cr/actions/workflows/examples.yml/badge.svg)](https://github.com/kojix2/llama.cr/actions/workflows/examples.yml)
[![docs](https://img.shields.io/badge/docs-latest-blue.svg)](https://kojix2.github.io/llama.cr)
[![Lines of Code](https://img.shields.io/endpoint?url=https%3A%2F%2Ftokei.kojix2.net%2Fbadge%2Fgithub%2Fkojix2%2Fllama.cr%2Flines)](https://tokei.kojix2.net/github/kojix2/llama.cr)
![Static Badge](https://img.shields.io/badge/PURE-VIBE_CODING-magenta)

Crystal bindings for [llama.cpp](https://github.com/ggml-org/llama.cpp), a C/C++ implementation of LLaMA, Falcon, GPT-2, and other large language models.

The version in `shard.yml` corresponds to the compatible llama.cpp build number.
For example, shard version `0.10809.1` targets llama.cpp build `b10809`, which
is the stable release [v0.4.0](https://github.com/ggml-org/llama.cpp/releases/tag/v0.4.0).

This project is under active development and may change rapidly.

## Features

- Low-level bindings to the llama.cpp C API
- High-level Crystal wrapper classes for easy usage
- Memory management for C resources
- Simple text generation interface
- Backend capability checks and configurable GPU offloading
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
- [Llama 3 8B Instruct](https://huggingface.co/mmnga/Meta-Llama-3-8B-Instruct-gguf)
- [Mistral 7B Instruct v0.2](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF)

## Usage

### Basic Text Generation

```crystal
require "llama"

response = Llama.generate(
  "/path/to/model.gguf",
  "Once upon a time",
  max_tokens: 100,
  temperature: 0.8
)
puts response
```

For a typed result with a finish reason, token usage, stop sequences, or
streaming, use the additive `Llama.complete` API:

```crystal
options = Llama::GenerationOptions.new(max_tokens: 100, stop: ["\n\n"])

result = Llama.complete(
  "/path/to/model.gguf",
  "Once upon a time",
  options
)
puts result.text
puts result.finish_reason
puts result.usage.tokens_per_second
```

`Llama.generate` continues to return a `String`. `Llama.complete` returns a
`Generation`; both use the same checked generation path.

### Streaming and Stateful Sessions

Use a `Session` to reuse one context. Chunks contain valid UTF-8 and stop text is
buffered so it is not emitted unless `include_stop` is enabled.

```crystal
Llama::Model.open("/path/to/model.gguf") do |model|
  model.session do |session|
    result = session.generate("Once upon a time") do |chunk|
      print chunk.text
      STDOUT.flush
    end
    puts "\n#{result.finish_reason}"
  end
end
```

`Session` keeps a canonical transcript between calls. Call `reset` to start a
new sequence. `snapshot`, `restore`, `save`, and `load` validate the model and
native version before changing that transcript. Only one generation may use a
session at a time.

### Backend Capabilities and GPU Offloading

GPU offloading depends on the linked llama.cpp build and the backends available
at runtime. Check capabilities before selecting accelerator-specific settings:

```crystal
puts Llama.gpu_offload_supported?
puts Llama.mmap_supported?
puts Llama.mlock_supported?
puts Llama.rpc_supported?
```

With GPU offloading available, the convenience API can offload the model and
context operations:

```crystal
raise "GPU offloading is unavailable" unless Llama.gpu_offload_supported?

response = Llama.generate(
  "/path/to/model.gguf",
  "Once upon a time",
  n_gpu_layers: -1,
  offload_kqv: true,
  op_offload: true
)
```

`n_gpu_layers: -1` requests all model layers; `0` keeps them on the CPU.
`offload_kqv` controls KQV operations and the KV cache, while `op_offload`
controls host tensor operations. These options do not add GPU support to a
CPU-only llama.cpp build. The defaults are `n_gpu_layers: 0`,
`offload_kqv: false`, and `op_offload: false`.

The same context settings are available when managing resources directly:

```crystal
Llama::Model.open("/path/to/model.gguf", n_gpu_layers: -1) do |model|
  model.context(offload_kqv: true, op_offload: true) do |context|
    puts context.generate("Once upon a time")
  end
end
```

### Lazy Model Loading

llama.cpp can load eligible model tensors on demand. The default is
`Llama::LazyMode::AUTO`, which lazily loads marked tensors larger than 4 GiB.

```crystal
Llama::Model.open("/path/to/model.gguf", lazy_mode: Llama::LazyMode::ON) do |model|
  model.context do |context|
    puts context.generate("Once upon a time")
  end
end
```

Use `Llama::LazyMode::OFF` to always read complete tensors up front.

The convenience API accepts the same setting:

```crystal
response = Llama.generate(
  "/path/to/model.gguf",
  "Once upon a time",
  lazy_mode: Llama::LazyMode::ON
)
```

### Resource Lifetime

Native-backed objects provide an idempotent `free` method. Prefer the block APIs
shown above for deterministic cleanup. For longer-lived resources, call `free`
in an `ensure` block and release dependencies before their owners:

```crystal
model = Llama::Model.new("/path/to/model.gguf")
context = model.context

begin
  puts context.generate("Once upon a time")
ensure
  context.free
  model.free
end
```

Release samplers and adapters before contexts, and contexts before models.
Samplers added to a `SamplerChain` are released with the chain.

### Backend Lifetime

`Llama.init` is called automatically when a model or context is created, so most
applications do not need to call it manually.

`Llama.uninit` is optional and usually not needed. It is intended only for
controlled teardown after all `Llama::Model` and `Llama::Context` instances have
been finalized. Calling it while models or contexts are still alive raises an
error, because their finalizers may still need the llama.cpp backend.

### Saved State Compatibility

llama.cpp b10809 updates the session and sequence-state file formats. Session
or state files written by b10566 are not guaranteed to load with this version;
recreate them after upgrading.

### Advanced Sampling

```crystal
require "llama"

Llama::Model.open("/path/to/model.gguf") do |model|
  model.context do |context|
    Llama::SamplerChain.open do |chain|
      chain.add(Llama::Sampler::TopK.new(40))
      chain.add(Llama::Sampler::MinP.new(0.05, 1))
      chain.add(Llama::Sampler::Temp.new(0.8))
      chain.add(Llama::Sampler::Dist.new(42))

      result = context.generate_with_sampler("Write a short poem about AI:", chain, 150)
      puts result
    end
  end
end
```

### Chat Conversations

```crystal
require "llama"

Llama::Model.open("/path/to/model.gguf") do |model|
  model.chat(system: "You are a helpful assistant.") do |chat|
    result = chat.ask("Hello, who are you?") do |chunk|
      print chunk.text
    end
    puts "\n#{result.finish_reason}"
  end
end
```

Chat history is committed transactionally. A cancelled turn is not committed
unless `commit_partial: true` is requested. b10809 recognizes predefined chat
template shapes; it is not a general Jinja evaluator. Pass a recognized template
explicitly when the model does not provide one.

### Embeddings

```crystal
require "llama"

Llama::Model.open("/path/to/model.gguf") do |model|
  model.embedder(pooling: Llama::Pooling::Mean) do |embedder|
    vector = embedder.embed("Hello, world!", normalize: true)
    vectors = embedder.embed_all(["one", "two"], normalize: true)
    puts "Embedding dimension: #{vector.size}"
  end
end
```

`Embedder` owns a dedicated embedding context, copies native vectors before the
next call, and preserves input order in native multi-sequence batches.

### Utilities

#### System Info

```crystal
info = Llama.runtime_info
puts "llama.cr #{info.wrapper_version} expects #{info.expected_build}"
puts "loaded llama.cpp #{info.reported_version} with #{info.backend_count} backends"
puts info.system_info
```

#### Tokenization Utility

```crystal
Llama::Model.open("/path/to/model.gguf", vocab_only: true) do |model|
  puts Llama.tokenize_and_format(model.vocab, "Hello, world!", ids_only: true)
end
```

## Examples

The `examples` directory contains sample code demonstrating various features:

- `simple.cr` - Basic text generation
- `minimal.cr` - Minimal use of the high-level generation API
- `chat.cr` - Chat conversations with models
- `streaming.cr` - UTF-8-safe streamed generation with a session
- `embedding.cr` - Single-batch normalized sentence embeddings
- `tokenize.cr` - Tokenization and vocabulary features
- `server.cr` - HTTP streaming server (uses `examples/shard.yml`)

## API Documentation

See [kojix2.github.io/llama.cr](https://kojix2.github.io/llama.cr) for full API docs.

### API Layers

- Convenience API: `Llama.generate` and `Context#generate` retain their existing
  string-returning behavior.
- Typed helpers: `GenerationOptions`, `Session`, `Chat`, `Embedder`, and
  `Sampling::Plan` add structured results and managed workflows.
- Advanced API: `Context`, `Batch`, `Memory`, `State`, and manual samplers expose
  native concepts. Borrowed views are valid only while their owner remains open;
  copy pointer-backed data before another native call.
- Raw API: `require "llama/raw"` exposes `Llama::LibLlama`. Its structs, symbols,
  and pointer lifetimes track the pinned upstream build and may change between
  shard releases.

The wrapper checks the reported stable version (`0.4.0`) before passing ABI-
sensitive structs by value. The C API does not report the exact build number, so
the exact `b10809` package pin and CI ABI checks remain required.

Custom `Llama.log_set` callbacks are experimental. On the pinned b10809 build,
model loading and multithreaded decode callbacks were observed on the calling
thread. Callback exceptions are contained at the C boundary and can be retrieved
with `Llama.take_log_callback_error`.

Implementation tradeoffs and deferred optimizations are recorded in
[IMPLEMENTATION_DECISIONS.md](IMPLEMENTATION_DECISIONS.md).

### Core Classes

- [Llama::Model](https://kojix2.github.io/llama.cr/Llama/Model.html) - Represents a loaded LLaMA model
- [Llama::Context](https://kojix2.github.io/llama.cr/Llama/Context.html) - Handles inference state for a model
- [Llama::Vocab](https://kojix2.github.io/llama.cr/Llama/Vocab.html) - Provides access to the model's vocabulary
- `Llama::Session` - Reusable typed and streaming generation
- `Llama::Chat` - Transactional conversation history
- `Llama::Embedder` - Safe single and batched sentence embeddings
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
