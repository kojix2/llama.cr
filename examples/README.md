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
LIBRARY_PATH=/path/to/llama/libs crystal build simple.cr --link-flags="-L/path/to/llama/libs -Wl,-rpath,/path/to/llama/libs -lllama -lggml"
LD_LIBRARY_PATH=/path/to/llama/libs ./simple --model /path/to/model.gguf "Once upon a time"
```

### Chat Example

This example demonstrates how to use the chat functionality with chat templates.

```bash
LIBRARY_PATH=/path/to/llama/libs crystal build chat.cr --link-flags="-L/path/to/llama/libs -Wl,-rpath,/path/to/llama/libs -lllama -lggml"
LD_LIBRARY_PATH=/path/to/llama/libs ./chat --model /path/to/model.gguf
```

### Tokenization Example

This example demonstrates how to tokenize text and work with the model's vocabulary.

```bash
LIBRARY_PATH=/path/to/llama/libs crystal build tokenize.cr --link-flags="-L/path/to/llama/libs -Wl,-rpath,/path/to/llama/libs -lllama -lggml"
LD_LIBRARY_PATH=/path/to/llama/libs ./tokenize --model /path/to/model.gguf --prompt "Hello, world!"
```

### Web Chat Server

This example starts a browser-based chat UI.

```bash
cd examples
shards install
LIBRARY_PATH=/path/to/llama/libs crystal build server.cr --link-flags="-L/path/to/llama/libs -Wl,-rpath,/path/to/llama/libs -lllama -lggml" -o server
LD_LIBRARY_PATH=/path/to/llama/libs ./server --model /path/to/model.gguf --port 3000
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
- If backend auto-detection fails, set `GGML_BACKEND_PATH` to a backend library file (for example `libggml-cpu-x64.so`) as a fallback
- `GGML_BACKEND_PATH` must point to a file, not a directory

Example:

```bash
LIBRARY_PATH=/path/to/llama/libs \
LD_LIBRARY_PATH=/path/to/llama/libs \
GGML_BACKEND_PATH=/path/to/llama/libs/libggml-cpu-x64.so \
./simple --model /path/to/model.gguf "Once upon a time"
```
