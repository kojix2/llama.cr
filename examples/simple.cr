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

# The backend is automatically initialized by Llama::Model and Llama::Context

# Load the model
# Llama.init will be called automatically
model = Llama::Model.new(model_path, n_gpu_layers: ngl)

vocab = model.vocab

# Tokenize the prompt
prompt_tokens = vocab.tokenize(prompt)

# Initialize the context
context = Llama::Context.new(
  model,
  n_ctx: (prompt_tokens.size + n_predict - 1).to_u32,
  n_batch: (prompt_tokens.size).to_u32
)

# Initialize the sampler
sampler = Llama::SamplerChain.new(no_perf: false)

# Add a greedy sampler to the chain
sampler.add(Llama::Sampler::Greedy.new)

# Print the prompt token-by-token
prompt_tokens.each do |token|
  piece = vocab.token_to_piece(token, 0, true)
  print piece
end

# Use the high-level generate method instead of manual batch processing
response = context.generate("", max_tokens: n_predict, temperature: 0.0)
print response

t_start = Time.monotonic
elapsed = Time.monotonic - t_start
n_decode = n_predict

puts

elapsed = Time.monotonic - t_start

STDERR.puts "Decoded #{n_decode} tokens in #{elapsed.total_seconds.round(2)} s, speed: #{(n_decode / elapsed.total_seconds).round(2)} t/s"

STDERR.puts
sampler.print_perf
context.print_perf
STDERR.puts

Llama.uninit
