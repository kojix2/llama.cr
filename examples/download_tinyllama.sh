#!/bin/bash
# Download TinyLlama-1.1B-Chat model (GGUF format) to ./models/tinyllama.gguf (relative to this script)
# Requires: curl or wget

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="$SCRIPT_DIR/models"
URL="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
OUT="$MODEL_DIR/tinyllama.gguf"

mkdir -p "$MODEL_DIR"

echo "Downloading to $OUT ..."

if command -v curl >/dev/null 2>&1; then
  curl -L "$URL" -o "$OUT"
elif command -v wget >/dev/null 2>&1; then
  wget -O "$OUT" "$URL"
else
  echo "Error: curl or wget required." >&2
  exit 1
fi

echo "Done."
