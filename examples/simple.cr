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

  parser.on("-n N", "--n-predict=N", "Number of tokens to predict (default: 32)") do |count|
    n_predict = count.to_i
  end

  parser.on("-g N", "--n-gpu-layers=N", "Number of layers to offload to GPU (default: 99)") do |layers|
    ngl = layers.to_i
  end

  parser.on("--ngl=N", "Alias for --n-gpu-layers") do |layers|
    ngl = layers.to_i
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

# Scope native resources so they are released in dependency order on every exit
# path. This is required by llama.cpp 0.2.0 (ggml 0.21.0) on Metal.
Llama::Model.open(model_path, n_gpu_layers: ngl) do |model|
  vocab = model.vocab
  prompt_tokens = vocab.tokenize(prompt)

  model.context(
    n_ctx: (prompt_tokens.size + n_predict - 1).to_u32,
    n_batch: prompt_tokens.size.to_u32
  ) do |context|
    Llama::SamplerChain.open(no_perf: false) do |sampler|
      sampler.add(Llama::Sampler::Greedy.new)

      elapsed = Time.measure do
        print prompt
        response = context.generate_with_sampler(prompt, sampler, n_predict)
        print response
      end

      puts

      STDERR.puts "Decoded #{n_predict} tokens in #{elapsed.total_seconds.round(2)} s, speed: #{(n_predict / elapsed.total_seconds).round(2)} t/s"

      STDERR.puts
      sampler.print_perf
      context.print_perf
      STDERR.puts
    end
  end
end
