require "../src/llama"
require "option_parser"
require "colorize"

# Parse command line arguments
model_path = ""
n_ctx = 2048
ngl = -1

OptionParser.parse do |parser|
  parser.banner = "Usage: #{PROGRAM_NAME} -m MODEL [-c context_size] [-ngl n_gpu_layers]"

  parser.on("-m", "--model MODEL", "Path to the model file (required)") do |path|
    model_path = path
  end

  parser.on("-c", "--context N", "Context size (default: 2048)") do |context_size|
    n_ctx = context_size.to_i
  end

  parser.on("-g", "--gpu-layers N", "Number of layers to offload to GPU (default: -1, all layers)") do |layers|
    ngl = layers.to_i
  end

  parser.on("-h", "--help", "Show this help") do
    puts parser
    exit
  end
end

abort "Error: Model path is required. Use -m or --model option.\nRun with --help for usage information." if model_path.empty?

Llama.log_level = Llama::LOG_LEVEL_ERROR

# Scope native resources so they are released in dependency order on every exit
# path. This is required by newer llama.cpp releases on Metal.
begin
  Llama::Model.open(model_path, n_gpu_layers: ngl) do |model|
    use_gpu = ngl != 0 && Llama.gpu_offload_supported?
    context_options = Llama::ContextOptions.new(
      context_size: n_ctx.to_u32,
      batch_size: n_ctx.to_u32,
      offload_kqv: use_gpu,
      op_offload: use_gpu
    )
    sampling = Llama::Sampling::Plan.new([
      Llama::Sampling::MinP.new(0.05_f32),
      Llama::Sampling::Temperature.new(0.8_f32),
      Llama::Sampling::Distribution.new,
    ] of Llama::Sampling::Stage)
    generation_options = Llama::GenerationOptions.new(sampling: sampling)

    model.chat(context_options: context_options) do |chat|
      loop do
        print "> ".colorize(:green)
        user_input = gets
        break if user_input.nil? || user_input.empty?

        print "Assistant: ".colorize(:yellow)
        result = chat.ask(user_input, generation_options) do |chunk|
          print chunk.text
          STDOUT.flush
        end
        puts "\n[#{result.finish_reason}]"
      end
    end
  end
rescue ex
  abort ex.message
end
