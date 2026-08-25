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

  parser.on("-c", "--context N", "Context size (default: 2048)") do |context_size|
    n_ctx = context_size.to_i
  end

  parser.on("-g", "--gpu-layers N", "Number of layers to offload to GPU (default: 99)") do |layers|
    ngl = layers.to_i
  end

  parser.on("-h", "--help", "Show this help") do
    puts parser
    exit
  end
end

abort "Error: Model path is required. Use -m or --model option.\nRun with --help for usage information." if model_path.empty?

Llama.log_level = Llama::LOG_LEVEL_ERROR

def generate(context, vocab, sampler, prompt) : String
  sampler.reset
  response = ""
  is_first = true
  prompt_tokens = vocab.tokenize(prompt, add_special: is_first, parse_special: true)

  if prompt_tokens.empty?
    STDERR.puts "Failed to tokenize the prompt"
    return response
  end

  batch = Llama::Batch.from_tokens(prompt_tokens)
  pos = prompt_tokens.size
  begin
    loop do
      n_ctx = context.n_ctx
      if batch.n_tokens > n_ctx
        raise Llama::Context::Error.new("Context size exceeded")
      end

      if context.decode(batch) != 0
        STDERR.puts "Failed to decode"
        break
      end
      batch.free

      new_token_id = sampler.sample(context)
      break if vocab.eog?(new_token_id)

      piece = vocab.token_to_piece(new_token_id, 0, true)
      print piece
      STDOUT.flush
      response += piece

      batch = Llama::Batch.from_tokens([new_token_id])
      batch.to_unsafe.pos[0] = pos
      pos += 1
    end
  ensure
    batch.free
  end
  response
end

# Scope native resources so they are released in dependency order on every exit
# path. This is required by llama.cpp 0.2.0 (ggml 0.21.0) on Metal.
begin
  Llama::Model.open(model_path, n_gpu_layers: ngl) do |model|
    vocab = model.vocab

    model.context(n_ctx: n_ctx.to_u32, n_batch: n_ctx.to_u32) do |context|
      Llama::SamplerChain.open do |sampler|
        sampler.add(Llama::Sampler::MinP.new(0.05, 1))
        sampler.add(Llama::Sampler::Temp.new(0.8))
        sampler.add(Llama::Sampler::Dist.new(Llama::DEFAULT_SEED))

        tmpl = model.chat_template
        if tmpl.nil?
          STDERR.puts "Warning: Model does not provide a chat template, using default"
          tmpl = ""
        end

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
      end
    end
  end
rescue ex
  abort ex.message
end
