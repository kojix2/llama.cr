#!/bin/bash
# Download llama.cpp headers from a specific version

# Get version information from shards
VERSION=$(cd "$(dirname "$0")/.." && shards version)
LLAMA_BUILD="b${VERSION}"

echo "Downloading llama.cpp headers version ${LLAMA_BUILD}..."

# Download URL
BASE_URL="https://raw.githubusercontent.com/ggml-org/llama.cpp/${LLAMA_BUILD}"

# Download header files
wget "${BASE_URL}/LICENSE" -O "$(dirname "$0")/LICENSE"
wget "${BASE_URL}/include/llama.h" -O "$(dirname "$0")/llama.h"
wget "${BASE_URL}/ggml/include/ggml.h" -O "$(dirname "$0")/ggml.h"
wget "${BASE_URL}/ggml/include/ggml-cpu.h" -O "$(dirname "$0")/ggml-cpu.h"
wget "${BASE_URL}/ggml/include/ggml-backend.h" -O "$(dirname "$0")/ggml-backend.h"
wget "${BASE_URL}/ggml/include/ggml-opt.h" -O "$(dirname "$0")/ggml-opt.h"

echo "Downloaded llama.cpp headers from version ${LLAMA_BUILD}"
