#!/usr/bin/env crystal

# Simple example using llama.cr
#
# This example demonstrates the basic usage of llama.cr with low-level batch processing
# and a simple greedy sampler. It's a direct port of the C++ example from llama.cpp.
#
# Compilation:
#   crystal build examples/simple.cr --link-flags="-L/path/to/llama.cpp/build/bin"
#
# Execution:
#   LD_LIBRARY_PATH=/path/to/llama.cpp/build/bin ./simple -m /path/to/model.gguf [-n n_predict] [-ngl n_gpu_layers] [prompt]

require "../src/llama"

require "option_parser"

# Parse command line arguments
model_path = ""
prompt = "Hello my name is"
ngl = 99
n_predict = 32

OptionParser.parse do |parser|
  parser.banner = "Usage: #{PROGRAM_NAME} -m MODEL [options] [prompt]"

  parser.on("-m MODEL", "--model=MODEL", "Path to the model file (required)") do |path|
    model_path = path
  end

  parser.on("-n N", "--n-predict=N", "Number of tokens to predict (default: 32)") do |n|
    n_predict = n.to_i
  end

  parser.on("-ngl N", "--n-gpu-layers=N", "Number of layers to offload to GPU (default: 99)") do |n|
    ngl = n.to_i
  end

  parser.on("-h", "--help", "Show this help") do
    puts parser
    exit
  end

  parser.unknown_args do |args|
    if args.size > 0
      prompt = args.join(" ")
    end
  end
end

if model_path.empty?
  STDERR.puts "Error: Model path is required. Use -m or --model option."
  STDERR.puts "Run with --help for usage information."
  exit(1)
end

# Initialize the backend
Llama::LibLlama.llama_backend_init

# Load the model
model_params = Llama::LibLlama.llama_model_default_params
model_params.n_gpu_layers = ngl

begin
  model = Llama::Model.new(model_path)
rescue ex
  STDERR.puts "Error: unable to load model: #{ex.message}"
  exit(1)
end

vocab = model.vocab

# Tokenize the prompt
begin
  prompt_tokens = vocab.tokenize(prompt)
rescue ex
  STDERR.puts "Error: failed to tokenize the prompt: #{ex.message}"
  exit(1)
end

if prompt_tokens.empty?
  STDERR.puts "Error: prompt tokenization resulted in empty token array"
  exit(1)
end

# Initialize the context
ctx_params = Llama::LibLlama.llama_context_default_params
# n_ctx is the context size
ctx_params.n_ctx = prompt_tokens.size + n_predict - 1
# n_batch is the maximum number of tokens that can be processed in a single call to llama_decode
ctx_params.n_batch = prompt_tokens.size
# enable performance counters
ctx_params.no_perf = false

begin
  context = model.context(ctx_params)
rescue ex
  STDERR.puts "Error: failed to create the context: #{ex.message}"
  exit(1)
end

# Initialize the sampler
sparams = Llama::LibLlama.llama_sampler_chain_default_params
sparams.no_perf = false
sampler = Llama::SamplerChain.new(sparams)

# Add a greedy sampler to the chain
sampler.add(Llama::GreedySampler.new)

# Print the prompt token-by-token
prompt_tokens.each do |token|
  begin
    piece = vocab.token_to_piece(token, 0, true)
    print piece
  rescue ex
    STDERR.puts "Error: failed to convert token to piece: #{ex.message}"
    exit(1)
  end
end

# Prepare a batch for the prompt
batch = Llama::Batch.for_tokens(prompt_tokens)

# Main loop
t_main_start = Llama::LibLlama.llama_time_us
n_decode = 0
new_token_id = 0

n_pos = 0
while n_pos + batch.n_tokens < prompt_tokens.size + n_predict
  # Evaluate the current batch with the transformer model
  begin
    result = context.decode(batch)
    if result != 0
      STDERR.puts "Error: failed to eval, return code #{result}"
      exit(1)
    end
  rescue ex
    STDERR.puts "Error: failed to decode batch: #{ex.message}"
    exit(1)
  end

  n_pos += batch.n_tokens

  # Sample the next token
  begin
    # Sample the next token using the sampler chain
    new_token_id = sampler.sample(context)

    # Is it an end of generation?
    if vocab.is_eog(new_token_id)
      break
    end

    # Convert the token to text and print it
    piece = vocab.token_to_piece(new_token_id, 0, true)
    print piece
    STDOUT.flush

    # Prepare the next batch with the sampled token
    batch = Llama::Batch.for_tokens([new_token_id])

    n_decode += 1
  rescue ex
    STDERR.puts "Error: sampling failed: #{ex.message}"
    exit(1)
  end
end

puts

t_main_end = Llama::LibLlama.llama_time_us
elapsed_time = (t_main_end - t_main_start) / 1_000_000.0

STDERR.puts "Decoded #{n_decode} tokens in #{elapsed_time.round(2)} s, speed: #{(n_decode / elapsed_time).round(2)} t/s"

STDERR.puts
sampler.print_perf
context.print_perf
STDERR.puts

# Resources will be automatically freed by the GC, but we can explicitly clean up
sampler.finalize
context.finalize
model.finalize
