require "../src/llama"
require "../src/llama/chat"
require "option_parser"
require "colorize"

# Parse command line arguments
model_path = ""
n_ctx = 2048
ngl = 99

OptionParser.parse do |parser|
  parser.banner = "Usage: #{PROGRAM_NAME} -m MODEL [-c context_size] [-ngl n_gpu_layers]"

  parser.on("-m", "--model MODEL", "Path to the model file (required)") do |path|
    model_path = path
  end

  parser.on("-c", "--context N", "Context size (default: 2048)") do |n|
    n_ctx = n.to_i
  end

  parser.on("-g", "--gpu-layers N", "Number of layers to offload to GPU (default: 99)") do |n|
    ngl = n.to_i
  end

  parser.on("-h", "--help", "Show this help") do
    puts parser
    exit
  end
end

abort "Error: Model path is required. Use -m or --model option.\nRun with --help for usage information." if model_path.empty?

Llama.log_level = Llama::LOG_LEVEL_ERROR

# Initialize model, context, and sampler
model = Llama::Model.new(model_path, n_gpu_layers: ngl) || abort "Error: Unable to load model"
vocab = model.vocab
context = model.context(n_ctx: n_ctx.to_u32, n_batch: n_ctx.to_u32) || abort "Error: Failed to create the context"

sampler = Llama::Sampler::Chain.new
sampler.add(Llama::Sampler::MinP.new(0.05, 1))
sampler.add(Llama::Sampler::Temp.new(0.8))
sampler.add(Llama::Sampler::Dist.new(Llama::DEFAULT_SEED))

tmpl = model.chat_template
if tmpl.nil?
  STDERR.puts "Warning: Model does not provide a chat template, using default"
  tmpl = ""
end

# Evaluate prompt and generate response
def generate(context, vocab, sampler, prompt) : String
  response = ""
  is_first = context.kv_cache.used_cells == 0
  prompt_tokens = vocab.tokenize(prompt, add_special: is_first, parse_special: true)

  if prompt_tokens.empty?
    STDERR.puts "Failed to tokenize the prompt"
    return response
  end

  batch = Llama::Batch.get_one(prompt_tokens)
  loop do
    n_ctx = context.n_ctx
    n_ctx_used = context.kv_cache.used_cells
    if n_ctx_used + batch.n_tokens > n_ctx
      puts
      abort "Context size exceeded"
    end

    if context.decode(batch) != 0
      STDERR.puts "Failed to decode"
      break
    end

    new_token_id = sampler.sample(context)
    break if vocab.is_eog(new_token_id) || new_token_id == vocab.eos || new_token_id == vocab.eot

    piece = vocab.token_to_piece(new_token_id, 0, true)
    print piece
    STDOUT.flush
    response += piece

    batch = Llama::Batch.get_one([new_token_id])
  end
  response
end

# Main chat loop
messages = [] of Llama::ChatMessage

loop do
  print "> ".colorize(:green)
  user_input = gets
  break if user_input.nil? || user_input.empty?

  messages << Llama::ChatMessage.new("user", user_input)
  prompt = context.apply_chat_template(messages, true, tmpl)

  print "".colorize(:yellow)
  response = generate(context, vocab, sampler, prompt)
  puts

  messages << Llama::ChatMessage.new("assistant", response)
end

Llama.uninit
