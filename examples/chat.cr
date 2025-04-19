require "../src/llama"
require "../src/llama/chat"
require "option_parser"

# ANSI color codes
USER_COLOR      = "\033[32m"
ASSISTANT_COLOR = "\033[33m"
RESET_COLOR     = "\033[0m"

# Parse command line arguments
model_path = ""
n_ctx = 2048
ngl = 99

OptionParser.parse do |parser|
  parser.banner = "Usage: #{PROGRAM_NAME} -m MODEL [-c context_size] [-ngl n_gpu_layers]"

  parser.on("-m MODEL", "--model=MODEL", "Path to the model file (required)") do |path|
    model_path = path
  end

  parser.on("-c N", "--context=N", "Context size (default: 2048)") do |n|
    n_ctx = n.to_i
  end

  parser.on("-ngl N", "--n-gpu-layers=N", "Number of layers to offload to GPU (default: 99)") do |n|
    ngl = n.to_i
  end

  parser.on("-h", "--help", "Show this help") do
    puts parser
    exit
  end
end

if model_path.empty?
  STDERR.puts "Error: Model path is required. Use -m or --model option."
  STDERR.puts "Run with --help for usage information."
  exit(1)
end

# Only print errors (match C++ behavior)
Llama::LibLlama.llama_log_set(
  ->(level : Int32, text : LibC::Char*, user_data : Void*) {
    if level >= 3 # GGML_LOG_LEVEL_ERROR = 3
      STDERR.print String.new(text)
    end
    nil # Return value for Void
  },
  nil
)

# Initialize the backend
Llama::LibLlama.llama_backend_init

# Initialize the model
model = Llama::Model.new(model_path, n_gpu_layers: ngl)
if model.nil?
  STDERR.puts "Error: Unable to load model"
  exit(1)
end

vocab = model.vocab

# Initialize the context
context = model.context(n_ctx: n_ctx.to_u32, n_batch: n_ctx.to_u32)
if context.nil?
  STDERR.puts "Error: Failed to create the context"
  exit(1)
end

# Initialize the sampler
# Use default params to match C++ implementation
sparams = Llama::LibLlama.llama_sampler_chain_default_params
smpl = Llama::Sampler::Chain.new(no_perf: sparams.no_perf)
smpl.add(Llama::Sampler::MinP.new(0.05, 1))
smpl.add(Llama::Sampler::Temp.new(0.8))
# Use LLAMA_DEFAULT_SEED explicitly to match C++ implementation
smpl.add(Llama::Sampler::Dist.new(Llama::LibLlama::LLAMA_DEFAULT_SEED))

# Helper function to evaluate a prompt and generate a response
def generate(context, vocab, smpl, prompt)
  response = ""

  # Check if this is the first batch (KV cache is empty)
  is_first = context.kv_cache.used_cells == 0

  # Tokenize the prompt
  prompt_tokens = vocab.tokenize(prompt, add_special: is_first, parse_special: true)

  if prompt_tokens.empty?
    STDERR.puts "Failed to tokenize the prompt"
    return response
  end

  # Prepare a batch for the prompt
  batch = Llama::Batch.get_one(prompt_tokens)
  new_token_id = 0

  loop do
    # Check if we have enough space in the context
    n_ctx = Llama::LibLlama.llama_n_ctx(context.to_unsafe)
    n_ctx_used = context.kv_cache.used_cells

    if n_ctx_used + batch.n_tokens > n_ctx
      puts "#{RESET_COLOR}"
      STDERR.puts "Context size exceeded"
      exit(0)
    end

    # Decode the batch
    if context.decode(batch) != 0
      STDERR.puts "Failed to decode"
      break
    end

    # Sample the next token
    new_token_id = smpl.sample(context)

    # Is it an end of generation?
    if vocab.is_eog(new_token_id) || new_token_id == vocab.eos || new_token_id == vocab.eot
      break
    end

    # Convert the token to a string, print it and add it to the response
    piece = vocab.token_to_piece(new_token_id, 0, true)
    print piece
    STDOUT.flush
    response += piece

    # Prepare the next batch with the sampled token
    batch = Llama::Batch.get_one([new_token_id])
  end

  return response
end

# Initialize chat messages array
messages = [] of Llama::ChatMessage

# Get the chat template
tmpl = model.chat_template
if tmpl.nil?
  STDERR.puts "Warning: Model does not provide a chat template, using default"
  # Use an empty string as the template if none is provided
  tmpl = ""
end

# Create a buffer for formatted messages
formatted = Array(Char).new(n_ctx)
prev_len = 0

# Main chat loop
loop do
  # Get user input
  print "#{USER_COLOR}> #{RESET_COLOR}"
  user_input = gets

  # Exit if user input is empty
  if user_input.nil? || user_input.empty?
    break
  end

  # Add the user input to the message list and format it
  messages << Llama::ChatMessage.new("user", user_input)

  # Apply the chat template
  # First get the required buffer size
  c_messages = messages.map(&.to_unsafe)
  new_len = Llama::LibLlama.llama_chat_apply_template(
    tmpl.to_unsafe,
    c_messages.to_unsafe,
    messages.size,
    true,
    nil,
    0
  )

  # Resize the buffer if needed
  if new_len > formatted.size
    formatted = Array(Char).new(new_len)
  end

  # Apply the template again with the properly sized buffer
  buffer = Pointer(LibC::Char).malloc(new_len)
  new_len = Llama::LibLlama.llama_chat_apply_template(
    tmpl.to_unsafe,
    c_messages.to_unsafe,
    messages.size,
    true,
    buffer,
    new_len
  )

  if new_len < 0
    STDERR.puts "Failed to apply the chat template"
    exit(1)
  end

  # Extract the prompt (only the new part since prev_len)
  prompt = String.new(buffer + prev_len, new_len - prev_len)

  # Generate a response
  print "#{ASSISTANT_COLOR}"
  response = generate(context, vocab, smpl, prompt)
  puts "\n#{RESET_COLOR}"

  # Add the response to the messages
  messages << Llama::ChatMessage.new("assistant", response)

  # Update the previous length for the next iteration
  prev_len = Llama::LibLlama.llama_chat_apply_template(
    tmpl.to_unsafe,
    messages.map(&.to_unsafe).to_unsafe,
    messages.size,
    false,
    nil,
    0
  )

  if prev_len < 0
    STDERR.puts "Failed to apply the chat template"
    exit(1)
  end
end

# Free resources
context.cleanup
model.cleanup
Llama::LibLlama.llama_backend_free
