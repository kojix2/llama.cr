# llama.cr Examples

This directory contains example programs demonstrating how to use the llama.cr library.

## Prerequisites

- Install [Crystal](https://crystal-lang.org/install/)
- Build [llama.cpp](https://github.com/ggml-org/llama.cpp) from source
- Download a GGUF model file

## Building llama.cpp

```bash
git clone https://github.com/ggml-org/llama.cpp.git
cd llama.cpp
mkdir build && cd build
cmake ..
cmake --build . --config Release
sudo cmake --install .
sudo ldconfig
```

## Downloading a Model

For testing, we recommend using a small model like TinyLlama:

```bash
curl -L https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -o tiny_model.gguf
```

For more models, visit [TheBloke's Hugging Face page](https://huggingface.co/TheBloke).

## Running the Examples

### Simple Text Generation

This example demonstrates how to generate text from a prompt.

```bash
crystal build simple.cr --link-flags="-L/path/to/llama.cpp/build/bin"
LD_LIBRARY_PATH=/path/to/llama.cpp/build/bin ./simple /path/to/model.gguf "Once upon a time"
```

### Chat Example

This example demonstrates how to use the chat functionality with chat templates.

```bash
crystal build chat.cr --link-flags="-L/path/to/llama.cpp/build/bin"
LD_LIBRARY_PATH=/path/to/llama.cpp/build/bin ./chat /path/to/model.gguf
```

### Tokenization Example

This example demonstrates how to tokenize text and work with the model's vocabulary.

```bash
crystal build tokenize.cr --link-flags="-L/path/to/llama.cpp/build/bin"
LD_LIBRARY_PATH=/path/to/llama.cpp/build/bin ./tokenize /path/to/model.gguf "Hello, world!"
```

## Example List

- `simple.cr` - Basic text generation
- `chat.cr` - Chat conversations with models
- `tokenize.cr` - Tokenization and vocabulary features

## Troubleshooting

### Library Not Found

If you get an error like `error while loading shared libraries: libllama.so: cannot open shared object file: No such file or directory`, make sure:

- The library was built correctly in llama.cpp
- You're using the correct path in the `LD_LIBRARY_PATH` environment variable
- The library file exists in the specified directory

### Compilation Errors

If you encounter compilation errors:

- Make sure you have the latest version of Crystal installed
- Ensure llama.cpp was built successfully
- Check that you're using the correct path with `--link-flags`

### Model Loading Errors

If the model fails to load:

- Verify the model file exists and is not corrupted
- Ensure you have enough RAM to load the model
- Try a smaller model if you're having memory issues
