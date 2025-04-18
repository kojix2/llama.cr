#!/bin/bash
# Download llama.cpp headers from a specific version

# Get version information from LLAMA_VERSION file
LLAMA_VERSION=$(cat "$(dirname "$0")/../LLAMA_VERSION")

echo "Downloading llama.cpp headers version ${LLAMA_VERSION}..."

# Download URL
BASE_URL="https://raw.githubusercontent.com/ggml-org/llama.cpp/${LLAMA_VERSION}"

# Download header files
wget "${BASE_URL}/LICENSE" -O "$(dirname "$0")/LICENSE"
wget "${BASE_URL}/include/llama.h" -O "$(dirname "$0")/llama.h"
wget "${BASE_URL}/ggml/include/ggml.h" -O "$(dirname "$0")/ggml.h"
wget "${BASE_URL}/ggml/include/ggml-cpu.h" -O "$(dirname "$0")/ggml-cpu.h"
wget "${BASE_URL}/ggml/include/ggml-backend.h" -O "$(dirname "$0")/ggml-backend.h"

echo "Downloaded llama.cpp headers from version ${LLAMA_VERSION}"
