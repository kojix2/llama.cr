require "file/tempfile"
require "../src/llama/lib_llama"

C_PROBE = <<-'C'
  #include <stddef.h>
  #include <stdio.h>
  #include "llama.h"

  #define PRINT_SIZE(type) printf(#type ".sizeof=%zu\\n", sizeof(struct type))
  #define PRINT_OFFSET(type, field) printf(#type "." #field "=%zu\\n", offsetof(struct type, field))

  int main(void) {
      PRINT_SIZE(llama_model_params);
      PRINT_OFFSET(llama_model_params, devices);
      PRINT_OFFSET(llama_model_params, tensor_buft_overrides);
      PRINT_OFFSET(llama_model_params, n_gpu_layers);
      PRINT_OFFSET(llama_model_params, split_mode);
      PRINT_OFFSET(llama_model_params, load_mode);
      PRINT_OFFSET(llama_model_params, main_gpu);
      PRINT_OFFSET(llama_model_params, tensor_split);
      PRINT_OFFSET(llama_model_params, progress_callback);
      PRINT_OFFSET(llama_model_params, progress_callback_user_data);
      PRINT_OFFSET(llama_model_params, kv_overrides);
      PRINT_OFFSET(llama_model_params, vocab_only);
      PRINT_OFFSET(llama_model_params, check_tensors);
      PRINT_OFFSET(llama_model_params, use_extra_bufts);
      PRINT_OFFSET(llama_model_params, no_host);
      PRINT_OFFSET(llama_model_params, no_alloc);
      PRINT_OFFSET(llama_model_params, load_mtp);

      PRINT_SIZE(llama_context_params);
      PRINT_OFFSET(llama_context_params, n_ctx);
      PRINT_OFFSET(llama_context_params, n_batch);
      PRINT_OFFSET(llama_context_params, n_ubatch);
      PRINT_OFFSET(llama_context_params, n_seq_max);
      PRINT_OFFSET(llama_context_params, n_rs_seq);
      PRINT_OFFSET(llama_context_params, n_outputs_max);
      PRINT_OFFSET(llama_context_params, n_threads);
      PRINT_OFFSET(llama_context_params, n_threads_batch);
      PRINT_OFFSET(llama_context_params, ctx_type);
      PRINT_OFFSET(llama_context_params, rope_scaling_type);
      PRINT_OFFSET(llama_context_params, pooling_type);
      PRINT_OFFSET(llama_context_params, attention_type);
      PRINT_OFFSET(llama_context_params, flash_attn_type);
      PRINT_OFFSET(llama_context_params, rope_freq_base);
      PRINT_OFFSET(llama_context_params, rope_freq_scale);
      PRINT_OFFSET(llama_context_params, yarn_ext_factor);
      PRINT_OFFSET(llama_context_params, yarn_attn_factor);
      PRINT_OFFSET(llama_context_params, yarn_beta_fast);
      PRINT_OFFSET(llama_context_params, yarn_beta_slow);
      PRINT_OFFSET(llama_context_params, yarn_orig_ctx);
      PRINT_OFFSET(llama_context_params, defrag_thold);
      PRINT_OFFSET(llama_context_params, cb_eval);
      PRINT_OFFSET(llama_context_params, cb_eval_user_data);
      PRINT_OFFSET(llama_context_params, type_k);
      PRINT_OFFSET(llama_context_params, type_v);
      PRINT_OFFSET(llama_context_params, abort_callback);
      PRINT_OFFSET(llama_context_params, abort_callback_data);
      PRINT_OFFSET(llama_context_params, embeddings);
      PRINT_OFFSET(llama_context_params, offload_kqv);
      PRINT_OFFSET(llama_context_params, no_perf);
      PRINT_OFFSET(llama_context_params, op_offload);
      PRINT_OFFSET(llama_context_params, swa_full);
      PRINT_OFFSET(llama_context_params, kv_unified);
      PRINT_OFFSET(llama_context_params, samplers);
      PRINT_OFFSET(llama_context_params, n_samplers);
      PRINT_OFFSET(llama_context_params, ctx_other);
      return 0;
  }
  C

def crystal_layout : Hash(String, Int32)
  {
    "llama_model_params.sizeof"                      => sizeof(Llama::LibLlama::LlamaModelParams),
    "llama_model_params.devices"                     => offsetof(Llama::LibLlama::LlamaModelParams, @devices),
    "llama_model_params.tensor_buft_overrides"       => offsetof(Llama::LibLlama::LlamaModelParams, @tensor_buft_overrides),
    "llama_model_params.n_gpu_layers"                => offsetof(Llama::LibLlama::LlamaModelParams, @n_gpu_layers),
    "llama_model_params.split_mode"                  => offsetof(Llama::LibLlama::LlamaModelParams, @split_mode),
    "llama_model_params.load_mode"                   => offsetof(Llama::LibLlama::LlamaModelParams, @load_mode),
    "llama_model_params.main_gpu"                    => offsetof(Llama::LibLlama::LlamaModelParams, @main_gpu),
    "llama_model_params.tensor_split"                => offsetof(Llama::LibLlama::LlamaModelParams, @tensor_split),
    "llama_model_params.progress_callback"           => offsetof(Llama::LibLlama::LlamaModelParams, @progress_callback),
    "llama_model_params.progress_callback_user_data" => offsetof(Llama::LibLlama::LlamaModelParams, @progress_callback_user_data),
    "llama_model_params.kv_overrides"                => offsetof(Llama::LibLlama::LlamaModelParams, @kv_overrides),
    "llama_model_params.vocab_only"                  => offsetof(Llama::LibLlama::LlamaModelParams, @vocab_only),
    "llama_model_params.check_tensors"               => offsetof(Llama::LibLlama::LlamaModelParams, @check_tensors),
    "llama_model_params.use_extra_bufts"             => offsetof(Llama::LibLlama::LlamaModelParams, @use_extra_bufts),
    "llama_model_params.no_host"                     => offsetof(Llama::LibLlama::LlamaModelParams, @no_host),
    "llama_model_params.no_alloc"                    => offsetof(Llama::LibLlama::LlamaModelParams, @no_alloc),
    "llama_model_params.load_mtp"                    => offsetof(Llama::LibLlama::LlamaModelParams, @load_mtp),
    "llama_context_params.sizeof"                    => sizeof(Llama::LibLlama::LlamaContextParams),
    "llama_context_params.n_ctx"                     => offsetof(Llama::LibLlama::LlamaContextParams, @n_ctx),
    "llama_context_params.n_batch"                   => offsetof(Llama::LibLlama::LlamaContextParams, @n_batch),
    "llama_context_params.n_ubatch"                  => offsetof(Llama::LibLlama::LlamaContextParams, @n_ubatch),
    "llama_context_params.n_seq_max"                 => offsetof(Llama::LibLlama::LlamaContextParams, @n_seq_max),
    "llama_context_params.n_rs_seq"                  => offsetof(Llama::LibLlama::LlamaContextParams, @n_rs_seq),
    "llama_context_params.n_outputs_max"             => offsetof(Llama::LibLlama::LlamaContextParams, @n_outputs_max),
    "llama_context_params.n_threads"                 => offsetof(Llama::LibLlama::LlamaContextParams, @n_threads),
    "llama_context_params.n_threads_batch"           => offsetof(Llama::LibLlama::LlamaContextParams, @n_threads_batch),
    "llama_context_params.ctx_type"                  => offsetof(Llama::LibLlama::LlamaContextParams, @ctx_type),
    "llama_context_params.rope_scaling_type"         => offsetof(Llama::LibLlama::LlamaContextParams, @rope_scaling_type),
    "llama_context_params.pooling_type"              => offsetof(Llama::LibLlama::LlamaContextParams, @pooling_type),
    "llama_context_params.attention_type"            => offsetof(Llama::LibLlama::LlamaContextParams, @attention_type),
    "llama_context_params.flash_attn_type"           => offsetof(Llama::LibLlama::LlamaContextParams, @flash_attn_type),
    "llama_context_params.rope_freq_base"            => offsetof(Llama::LibLlama::LlamaContextParams, @rope_freq_base),
    "llama_context_params.rope_freq_scale"           => offsetof(Llama::LibLlama::LlamaContextParams, @rope_freq_scale),
    "llama_context_params.yarn_ext_factor"           => offsetof(Llama::LibLlama::LlamaContextParams, @yarn_ext_factor),
    "llama_context_params.yarn_attn_factor"          => offsetof(Llama::LibLlama::LlamaContextParams, @yarn_attn_factor),
    "llama_context_params.yarn_beta_fast"            => offsetof(Llama::LibLlama::LlamaContextParams, @yarn_beta_fast),
    "llama_context_params.yarn_beta_slow"            => offsetof(Llama::LibLlama::LlamaContextParams, @yarn_beta_slow),
    "llama_context_params.yarn_orig_ctx"             => offsetof(Llama::LibLlama::LlamaContextParams, @yarn_orig_ctx),
    "llama_context_params.defrag_thold"              => offsetof(Llama::LibLlama::LlamaContextParams, @defrag_thold),
    "llama_context_params.cb_eval"                   => offsetof(Llama::LibLlama::LlamaContextParams, @cb_eval),
    "llama_context_params.cb_eval_user_data"         => offsetof(Llama::LibLlama::LlamaContextParams, @cb_eval_user_data),
    "llama_context_params.type_k"                    => offsetof(Llama::LibLlama::LlamaContextParams, @type_k),
    "llama_context_params.type_v"                    => offsetof(Llama::LibLlama::LlamaContextParams, @type_v),
    "llama_context_params.abort_callback"            => offsetof(Llama::LibLlama::LlamaContextParams, @abort_callback),
    "llama_context_params.abort_callback_data"       => offsetof(Llama::LibLlama::LlamaContextParams, @abort_callback_data),
    "llama_context_params.embeddings"                => offsetof(Llama::LibLlama::LlamaContextParams, @embeddings),
    "llama_context_params.offload_kqv"               => offsetof(Llama::LibLlama::LlamaContextParams, @offload_kqv),
    "llama_context_params.no_perf"                   => offsetof(Llama::LibLlama::LlamaContextParams, @no_perf),
    "llama_context_params.op_offload"                => offsetof(Llama::LibLlama::LlamaContextParams, @op_offload),
    "llama_context_params.swa_full"                  => offsetof(Llama::LibLlama::LlamaContextParams, @swa_full),
    "llama_context_params.kv_unified"                => offsetof(Llama::LibLlama::LlamaContextParams, @kv_unified),
    "llama_context_params.samplers"                  => offsetof(Llama::LibLlama::LlamaContextParams, @samplers),
    "llama_context_params.n_samplers"                => offsetof(Llama::LibLlama::LlamaContextParams, @n_samplers),
    "llama_context_params.ctx_other"                 => offsetof(Llama::LibLlama::LlamaContextParams, @ctx_other),
  }
end

def compile_probe(source_path : String, executable_path : String, assets_dir : String) : Nil
  compiler = ENV.fetch("CC", "cc")
  error = IO::Memory.new
  status = Process.run(compiler, ["-std=c11", "-I#{assets_dir}", source_path, "-o", executable_path], error: error)
  return if status.success?

  STDERR.print error.to_s
  raise "Failed to compile C ABI probe with #{compiler}"
end

def run_probe(executable_path : String) : Hash(String, Int32)
  output = IO::Memory.new
  error = IO::Memory.new
  status = Process.run(executable_path, output: output, error: error)
  unless status.success?
    STDERR.print error.to_s
    raise "C ABI probe failed"
  end

  output.to_s.lines.to_h do |line|
    key, value = line.split('=', 2)
    {key, value.to_i32}
  end
end

assets_dir = File.expand_path("../assets", __DIR__)
source = File.tempfile("llama-abi", ".c")
executable_path = File.tempname("llama-abi")

begin
  source.print(C_PROBE)
  source.close
  compile_probe(source.path, executable_path, assets_dir)

  c_layout = run_probe(executable_path)
  crystal_layout.each do |key, crystal_value|
    c_value = c_layout[key]?
    if c_value != crystal_value
      STDERR.puts "ABI mismatch for #{key}: C=#{c_value || "missing"}, Crystal=#{crystal_value}"
      exit 1
    end
  end
ensure
  source.close
  source.delete
  File.delete?(executable_path)
end

puts "ABI layouts match for llama_model_params and llama_context_params"
