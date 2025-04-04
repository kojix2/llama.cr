# llama.cr Examples

This directory contains example programs demonstrating how to use the llama.cr library.

## Prerequisites

Before running these examples, you need to:

1. Install Crystal (https://crystal-lang.org/install/)
2. Build llama.cpp from source
3. Download a GGUF model file

## Building llama.cpp

```bash
# Clone the repository
git clone https://github.com/ggml-org/llama.cpp.git

# Navigate to the directory
cd llama.cpp

# Create a build directory
mkdir build && cd build

# Configure with CMake
cmake ..

# Build (adjust -j4 to match your CPU cores)
cmake --build . --config Release -j4
```

The compiled library will be in the `build/bin` directory (or sometimes just `build` depending on your system).

## Downloading a Model

For testing, we recommend using a small model like TinyLlama:

```bash
# Download TinyLlama (about 600MB)
curl -L https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -o tiny_model.gguf
```

For more models, visit [TheBloke's Hugging Face page](https://huggingface.co/TheBloke).

## Running the Examples

### Text Generation Example

This example demonstrates how to generate text from a prompt.

```bash
# Compile the example (specify the path to llama.cpp build directory)
crystal build text_generation.cr --link-flags="-L/path/to/llama.cpp/build/bin"

# Run the example (specify the path to the library directory)
LD_LIBRARY_PATH=/path/to/llama.cpp/build/bin ./text_generation /path/to/model.gguf "Once upon a time"
```

Optional parameters:

- 3rd argument: Maximum number of tokens to generate (default: 128)
- 4th argument: Temperature parameter (default: 0.8)

Example with all parameters:

```bash
LD_LIBRARY_PATH=/path/to/llama.cpp/build/bin ./text_generation tiny_model.gguf "Once upon a time" 50 0.5
```

### Chat Example

This example demonstrates how to use the chat functionality with chat templates.

```bash
# Compile the example
crystal build chat.cr --link-flags="-L/path/to/llama.cpp/build/bin"

# Run the example
LD_LIBRARY_PATH=/path/to/llama.cpp/build/bin ./chat /path/to/model.gguf
```

### Sampling Example

This example demonstrates how to use different sampling methods.

```bash
# Compile the example
crystal build sampling.cr --link-flags="-L/path/to/llama.cpp/build/bin"

# Run the example
LD_LIBRARY_PATH=/path/to/llama.cpp/build/bin ./sampling /path/to/model.gguf "Once upon a time"
```

### Tokenization Example

This example demonstrates how to tokenize text and work with the model's vocabulary.

```bash
# Compile the example
crystal build tokenization.cr --link-flags="-L/path/to/llama.cpp/build/bin"

# Run the example
LD_LIBRARY_PATH=/path/to/llama.cpp/build/bin ./tokenization /path/to/model.gguf "Hello, world!"
```

### Advanced Features Example

This example demonstrates advanced features like KV cache management, batch processing, and state management.

```bash
# Compile the example
crystal build advanced.cr --link-flags="-L/path/to/llama.cpp/build/bin"

# Run the example
LD_LIBRARY_PATH=/path/to/llama.cpp/build/bin ./advanced /path/to/model.gguf
```

### Embeddings Example

This example demonstrates how to use the embeddings functionality to extract embeddings from text and calculate similarity between texts.

```bash
# Compile the example
crystal build embeddings.cr --link-flags="-L/path/to/llama.cpp/build/bin"

# Run the example
LD_LIBRARY_PATH=/path/to/llama.cpp/build/bin ./embeddings /path/to/model.gguf
```

### Advanced Sampling Methods Example

This example demonstrates how to use advanced sampling methods like Min-P, Typical, Mirostat, Grammar-based, and Penalties sampling.

```bash
# Compile the example
crystal build advanced_sampling.cr --link-flags="-L/path/to/llama.cpp/build/bin"

# Run the example
LD_LIBRARY_PATH=/path/to/llama.cpp/build/bin ./advanced_sampling /path/to/model.gguf
```

### Encoder-Decoder Models Example

This example demonstrates how to use encoder-decoder models like T5 or BART for sequence-to-sequence tasks such as translation.

```bash
# Compile the example
crystal build encoder_decoder.cr --link-flags="-L/path/to/llama.cpp/build/bin"

# Run the example
LD_LIBRARY_PATH=/path/to/llama.cpp/build/bin ./encoder_decoder --model=/path/to/model.gguf
```

Note: This example requires an encoder-decoder model like T5 or BART.

### New Advanced Sampling Methods Example

This example demonstrates how to use the newest sampling methods added to llama.cr, including Extended Temperature, Top-N Sigma, XTC, Infill, and Grammar Lazy Patterns sampling.

```bash
# Compile the example
crystal build advanced_sampling_methods.cr --link-flags="-L/path/to/llama.cpp/build/bin"

# Run the example
LD_LIBRARY_PATH=/path/to/llama.cpp/build/bin ./advanced_sampling_methods --model=/path/to/model.gguf
```

### Model Metadata Example

This example demonstrates how to access and use model metadata, including getting metadata values by key, listing all metadata entries, and getting model description.

```bash
# Compile the example
crystal build model_metadata.cr --link-flags="-L/path/to/llama.cpp/build/bin"

# Run the example
LD_LIBRARY_PATH=/path/to/llama.cpp/build/bin ./model_metadata /path/to/model.gguf
```

## Troubleshooting

### Library Not Found

If you get an error like `error while loading shared libraries: libllama.so: cannot open shared object file: No such file or directory`, make sure:

1. The library was built correctly in llama.cpp
2. You're using the correct path in the `LD_LIBRARY_PATH` environment variable
3. The library file exists in the specified directory

### Compilation Errors

If you encounter compilation errors:

1. Make sure you have the latest version of Crystal installed
2. Ensure llama.cpp was built successfully
3. Check that you're using the correct path with `--link-flags`

### Model Loading Errors

If the model fails to load:

1. Verify the model file exists and is not corrupted
2. Ensure you have enough RAM to load the model
3. Try a smaller model if you're having memory issues
