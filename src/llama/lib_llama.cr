module Llama
  {% if env("LLAMA_CPP_DIR") %}
    @[Link(ldflags: "-L `echo $LLAMA_CPP_DIR/build/bin` -lllama -lggml -Wl,-rpath,`echo $LLAMA_CPP_DIR/build/bin`")]
  {% else %}
    @[Link("llama")]
    @[Link("ggml")]
  {% end %}
  lib LibLlama
    # Constants
    LLAMA_DEFAULT_SEED      = 0xFFFFFFFF_u32
    LLAMA_TOKEN_NULL        =             -1
    LLAMA_FILE_MAGIC_GGLA   = 0x67676c61_u32 # 'ggla'
    LLAMA_FILE_MAGIC_GGSN   = 0x6767736e_u32 # 'ggsn'
    LLAMA_FILE_MAGIC_GGSQ   = 0x67677371_u32 # 'ggsq'
    LLAMA_SESSION_MAGIC     = LLAMA_FILE_MAGIC_GGSN
    LLAMA_SESSION_VERSION   = 9
    LLAMA_STATE_SEQ_MAGIC   = LLAMA_FILE_MAGIC_GGSQ
    LLAMA_STATE_SEQ_VERSION = 2

    alias LlamaToken = Int32
    alias LlamaPos = Int32
    alias LlamaSeqId = Int32

    type LlamaVocab = Void*
    type LlamaModel = Void*
    type LlamaContext = Void*
    type LlamaSampler = Void*
    type LlamaAdapterLora = Void*

    enum LlamaVocabType
      NONE = 0
      SPM  = 1
      BPE  = 2
      WPM  = 3
      UGM  = 4
      RWKV = 5
    end

    struct LlamaModelQuantizeParams
      nthread : Int32
      ftype : LlamaFtype
      output_tensor_type : Int32
      token_embedding_type : Int32
      allow_requantize : Bool
      quantize_output_tensor : Bool
      only_copy : Bool
      pure : Bool
      keep_split : Bool
      imatrix : Void*
      kv_overrides : Void*
      tensor_types : Void*
    end

    fun llama_model_quantize_default_params : LlamaModelQuantizeParams

    enum LlamaTokenAttr
      UNDEFINED    = 0
      UNKNOWN      = 1 << 0
      UNUSED       = 1 << 1
      NORMAL       = 1 << 2
      CONTROL      = 1 << 3
      USER_DEFINED = 1 << 4
      BYTE         = 1 << 5
      NORMALIZED   = 1 << 6
      LSTRIP       = 1 << 7
      RSTRIP       = 1 << 8
      SINGLE_WORD  = 1 << 9
    end

    enum LlamaVocabPreType
      DEFAULT        =  0
      LLAMA3         =  1
      DEEPSEEK_LLM   =  2
      DEEPSEEK_CODER =  3
      FALCON         =  4
      MPT            =  5
      STARCODER      =  6
      GPT2           =  7
      REFACT         =  8
      COMMAND_R      =  9
      STABLELM2      = 10
      QWEN2          = 11
      OLMO           = 12
      DBRX           = 13
      SMAUG          = 14
      PORO           = 15
      CHATGLM3       = 16
      CHATGLM4       = 17
      VIKING         = 18
      JAIS           = 19
      TEKKEN         = 20
      SMOLLM         = 21
      CODESHELL      = 22
      BLOOM          = 23
      GPT3_FINNISH   = 24
      EXAONE         = 25
      CHAMELEON      = 26
      MINERVA        = 27
      DEEPSEEK3_LLM  = 28
      GPT4O          = 29
      SUPERBPE       = 30
      TRILLION       = 31
      BAILINGMOE     = 32
      LLAMA4         = 33
      PIXTRAL        = 34
    end

    enum LlamaTokenType
      UNDEFINED    = 0
      NORMAL       = 1
      UNKNOWN      = 2
      CONTROL      = 3
      USER_DEFINED = 4
      UNUSED       = 5
      BYTE         = 6
    end

    struct LlamaModelTensorBuftOverride
      pattern : LibC::Char*
      buft : Int32
    end

    struct LlamaLogitBias
      token : LlamaToken
      bias : Float32
    end

    # Enum Definitions
    enum LlamaFtype
      ALL_F32        =    0
      MOSTLY_F16     =    1
      MOSTLY_Q4_0    =    2
      MOSTLY_Q4_1    =    3
      MOSTLY_Q8_0    =    7
      MOSTLY_Q5_0    =    8
      MOSTLY_Q5_1    =    9
      MOSTLY_Q2_K    =   10
      MOSTLY_Q3_K_S  =   11
      MOSTLY_Q3_K_M  =   12
      MOSTLY_Q3_K_L  =   13
      MOSTLY_Q4_K_S  =   14
      MOSTLY_Q4_K_M  =   15
      MOSTLY_Q5_K_S  =   16
      MOSTLY_Q5_K_M  =   17
      MOSTLY_Q6_K    =   18
      MOSTLY_IQ2_XXS =   19
      MOSTLY_IQ2_XS  =   20
      MOSTLY_Q2_K_S  =   21
      MOSTLY_IQ3_XS  =   22
      MOSTLY_IQ3_XXS =   23
      MOSTLY_IQ1_S   =   24
      MOSTLY_IQ4_NL  =   25
      MOSTLY_IQ3_S   =   26
      MOSTLY_IQ3_M   =   27
      MOSTLY_IQ2_S   =   28
      MOSTLY_IQ2_M   =   29
      MOSTLY_IQ4_XS  =   30
      MOSTLY_IQ1_M   =   31
      MOSTLY_BF16    =   32
      MOSTLY_TQ1_0   =   36
      MOSTLY_TQ2_0   =   37
      GUESSED        = 1024
    end

    enum LlamaRopeType
      NONE   = -1
      NORM   =  0
      NEOX   =  1
      MROPE  =  2
      VISION =  3
    end

    enum LlamaRopeScalingType
      UNSPECIFIED = -1
      NONE        =  0
      LINEAR      =  1
      YARN        =  2
      LONGROPE    =  3
      MAX_VALUE   = LONGROPE
    end

    enum LlamaPoolingType
      UNSPECIFIED = -1
      NONE        =  0
      MEAN        =  1
      CLS         =  2
      LAST        =  3
      RANK        =  4
    end

    enum LlamaAttentionType
      UNSPECIFIED = -1
      CAUSAL      =  0
      NON_CAUSAL  =  1
    end

    enum LlamaSplitMode
      NONE  = 0
      LAYER = 1
      ROW   = 2
    end

    enum LlamaModelKvOverrideType
      INT
      FLOAT
      BOOL
      STR
    end

    struct LlamaModelKvOverride
      tag : LlamaModelKvOverrideType
      key : LibC::Char[128]
      val_i64 : Int64
      val_f64 : Float64
      val_bool : Bool
      val_str : LibC::Char[128]
    end

    struct LlamaTokenData
      id : LlamaToken
      logit : Float32
      p : Float32
    end

    struct LlamaTokenDataArray
      data : LlamaTokenData*
      size : LibC::SizeT
      selected : Int64
      sorted : Bool
    end

    struct LlamaBatch
      n_tokens : Int32
      token : LlamaToken*
      embd : Float32*
      pos : LlamaPos*
      n_seq_id : Int32*
      seq_id : LlamaSeqId**
      logits : Int8*
    end

    struct LlamaChatMessage
      role : LibC::Char*
      content : LibC::Char*
    end

    # Adapter Functions
    fun llama_adapter_lora_init(model : LlamaModel*, path_lora : LibC::Char*) : LlamaAdapterLora*
    fun llama_adapter_lora_free(adapter : LlamaAdapterLora*) : Void
    fun llama_set_adapter_lora(ctx : LlamaContext*, adapter : LlamaAdapterLora*, scale : Float32) : Int32
    fun llama_rm_adapter_lora(ctx : LlamaContext*, adapter : LlamaAdapterLora*) : Int32
    fun llama_clear_adapter_lora(ctx : LlamaContext*) : Void
    fun llama_apply_adapter_cvec(ctx : LlamaContext*, data : Float32*, len : LibC::SizeT, n_embd : Int32, il_start : Int32, il_end : Int32) : Int32

    # Chat Functions
    fun llama_chat_apply_template(
      tmpl : LibC::Char*,
      chat : LlamaChatMessage*,
      n_msg : LibC::SizeT,
      add_ass : Bool,
      buf : LibC::Char*,
      length : Int32,
    ) : Int32
    fun llama_model_chat_template(model : LlamaModel*, name : LibC::Char*) : LibC::Char*
    fun llama_chat_builtin_templates(output : LibC::Char**, len : LibC::SizeT) : Int32

    struct LlamaSamplerChainParams
      no_perf : Bool
    end

    struct LlamaModelParams
      devices : Void*
      tensor_buft_overrides : LlamaModelTensorBuftOverride*
      n_gpu_layers : Int32
      split_mode : LlamaSplitMode
      main_gpu : Int32
      tensor_split : Float32*
      progress_callback : Void*
      progress_callback_user_data : Void*
      kv_overrides : LlamaModelKvOverride*
      vocab_only : Bool
      use_mmap : Bool
      use_mlock : Bool
      check_tensors : Bool
    end

    struct LlamaContextParams
      n_ctx : UInt32
      n_batch : UInt32
      n_ubatch : UInt32
      n_seq_max : UInt32
      n_threads : Int32
      n_threads_batch : Int32
      rope_scaling_type : LlamaRopeScalingType
      pooling_type : LlamaPoolingType
      attention_type : LlamaAttentionType
      rope_freq_base : Float32
      rope_freq_scale : Float32
      yarn_ext_factor : Float32
      yarn_attn_factor : Float32
      yarn_beta_fast : Float32
      yarn_beta_slow : Float32
      yarn_orig_ctx : UInt32
      defrag_thold : Float32
      cb_eval : Void*
      cb_eval_user_data : Void*
      type_k : Int32
      type_v : Int32
      abort_callback : Void*
      abort_callback_data : Void*
      embeddings : Bool
      offload_kqv : Bool
      flash_attn : Bool
      no_perf : Bool
      op_offload : Bool
      swa_full : Bool
    end

    fun llama_model_default_params : LlamaModelParams
    fun llama_context_default_params : LlamaContextParams
    fun llama_sampler_chain_default_params : LlamaSamplerChainParams

    # Initialization and Finalization
    fun llama_backend_init : Void
    fun llama_backend_free : Void
    fun llama_numa_init(numa : Int32) : Void
    fun llama_attach_threadpool(ctx : LlamaContext*, threadpool : Void*, threadpool_batch : Void*) : Void
    fun llama_detach_threadpool(ctx : LlamaContext*) : Void

    # Backend loading (required for newer llama.cpp versions)
    fun ggml_backend_load_all : Void

    # Backend verification
    fun ggml_backend_reg_count : LibC::SizeT

    # Memory API (modern replacement for deprecated KV cache API)
    alias LlamaMemoryT = Void*
    fun llama_get_memory(ctx : LlamaContext*) : LlamaMemoryT
    fun llama_memory_clear(mem : LlamaMemoryT, data : Bool) : Void
    fun llama_memory_seq_rm(mem : LlamaMemoryT, seq_id : LlamaSeqId, p0 : LlamaPos, p1 : LlamaPos) : Bool
    fun llama_memory_seq_cp(mem : LlamaMemoryT, seq_id_src : LlamaSeqId, seq_id_dst : LlamaSeqId, p0 : LlamaPos, p1 : LlamaPos) : Void
    fun llama_memory_seq_keep(mem : LlamaMemoryT, seq_id : LlamaSeqId) : Void
    fun llama_memory_seq_add(mem : LlamaMemoryT, seq_id : LlamaSeqId, p0 : LlamaPos, p1 : LlamaPos, delta : LlamaPos) : Void
    fun llama_memory_seq_div(mem : LlamaMemoryT, seq_id : LlamaSeqId, p0 : LlamaPos, p1 : LlamaPos, d : Int32) : Void
    fun llama_memory_seq_pos_min(mem : LlamaMemoryT, seq_id : LlamaSeqId) : LlamaPos
    fun llama_memory_seq_pos_max(mem : LlamaMemoryT, seq_id : LlamaSeqId) : LlamaPos
    fun llama_memory_can_shift(mem : LlamaMemoryT) : Bool

    # Model Functions
    fun llama_model_load_from_file(path_model : LibC::Char*, params : LlamaModelParams) : LlamaModel*
    fun llama_model_load_from_splits(paths : LibC::Char**, n_paths : LibC::SizeT, params : LlamaModelParams) : LlamaModel*
    fun llama_model_save_to_file(model : LlamaModel*, path_model : LibC::Char*) : Void
    fun llama_model_free(model : LlamaModel*) : Void
    fun llama_max_devices : LibC::SizeT
    fun llama_model_n_params(model : LlamaModel*) : UInt64
    fun llama_model_size(model : LlamaModel*) : UInt64
    fun llama_supports_mmap : Bool
    fun llama_supports_mlock : Bool
    fun llama_supports_gpu_offload : Bool
    fun llama_supports_rpc : Bool
    fun llama_model_n_ctx_train(model : LlamaModel*) : Int32
    fun llama_model_n_embd(model : LlamaModel*) : Int32
    fun llama_model_n_layer(model : LlamaModel*) : Int32
    fun llama_model_rope_type(model : LlamaModel*) : LlamaRopeType
    fun llama_model_n_head(model : LlamaModel*) : Int32
    fun llama_model_n_head_kv(model : LlamaModel*) : Int32
    fun llama_model_has_encoder(model : LlamaModel*) : Bool
    fun llama_model_has_decoder(model : LlamaModel*) : Bool
    fun llama_model_is_recurrent(model : LlamaModel*) : Bool
    fun llama_model_rope_freq_scale_train(model : LlamaModel*) : Float32
    fun llama_model_decoder_start_token(model : LlamaModel*) : LlamaToken
    fun llama_model_n_cls_out(model : LlamaModel*) : UInt32
    fun llama_model_cls_label(model : LlamaModel*, i : UInt32) : LibC::Char*
    fun llama_model_get_vocab(model : LlamaModel*) : LlamaVocab*

    # Model Metadata Functions
    fun llama_model_meta_val_str(model : LlamaModel*, key : LibC::Char*, buf : LibC::Char*, buf_size : LibC::SizeT) : Int32
    fun llama_model_meta_count(model : LlamaModel*) : Int32
    fun llama_model_meta_key_by_index(model : LlamaModel*, i : Int32, buf : LibC::Char*, buf_size : LibC::SizeT) : Int32
    fun llama_model_meta_val_str_by_index(model : LlamaModel*, i : Int32, buf : LibC::Char*, buf_size : LibC::SizeT) : Int32
    fun llama_model_desc(model : LlamaModel*, buf : LibC::Char*, buf_size : LibC::SizeT) : Int32

    # State Functions (current API)
    fun llama_state_get_size(ctx : LlamaContext*) : LibC::SizeT
    fun llama_state_get_data(ctx : LlamaContext*, dst : UInt8*, size : LibC::SizeT) : LibC::SizeT
    fun llama_state_set_data(ctx : LlamaContext*, src : UInt8*, size : LibC::SizeT) : LibC::SizeT
    fun llama_state_load_file(ctx : LlamaContext*, path_session : LibC::Char*, tokens_out : LlamaToken*, n_token_capacity : LibC::SizeT, n_token_count_out : LibC::SizeT*) : Bool
    fun llama_state_save_file(ctx : LlamaContext*, path_session : LibC::Char*, tokens : LlamaToken*, n_token_count : LibC::SizeT) : Bool

    # State Sequence Functions (additional API for specific sequences)
    fun llama_state_seq_get_size(ctx : LlamaContext*, seq_id : LlamaSeqId) : LibC::SizeT
    fun llama_state_seq_get_data(ctx : LlamaContext*, dst : UInt8*, size : LibC::SizeT, seq_id : LlamaSeqId) : LibC::SizeT
    fun llama_state_seq_set_data(ctx : LlamaContext*, src : UInt8*, size : LibC::SizeT, dest_seq_id : LlamaSeqId) : LibC::SizeT
    fun llama_state_seq_save_file(ctx : LlamaContext*, filepath : LibC::Char*, seq_id : LlamaSeqId, tokens : LlamaToken*, n_token_count : LibC::SizeT) : LibC::SizeT
    fun llama_state_seq_load_file(ctx : LlamaContext*, filepath : LibC::Char*, dest_seq_id : LlamaSeqId, tokens_out : LlamaToken*, n_token_capacity : LibC::SizeT, n_token_count_out : LibC::SizeT*) : LibC::SizeT

    # Context Functions
    fun llama_init_from_model(model : LlamaModel*, params : LlamaContextParams) : LlamaContext*
    fun llama_free(ctx : LlamaContext*)
    fun llama_get_model(ctx : LlamaContext*) : LlamaModel*
    fun llama_n_ctx(ctx : LlamaContext*) : UInt32
    fun llama_n_batch(ctx : LlamaContext*) : UInt32
    fun llama_n_ubatch(ctx : LlamaContext*) : UInt32
    fun llama_n_seq_max(ctx : LlamaContext*) : UInt32
    fun llama_encode(ctx : LlamaContext*, batch : LlamaBatch) : Int32
    fun llama_decode(ctx : LlamaContext*, batch : LlamaBatch) : Int32
    fun llama_get_logits(ctx : LlamaContext*) : Float32*
    fun llama_get_logits_ith(ctx : LlamaContext*, i : Int32) : Float32*
    fun llama_set_n_threads(ctx : LlamaContext*, n_threads : Int32, n_threads_batch : Int32) : Void
    fun llama_n_threads(ctx : LlamaContext*) : Int32
    fun llama_n_threads_batch(ctx : LlamaContext*) : Int32
    fun llama_set_embeddings(ctx : LlamaContext*, embeddings : Bool) : Void
    fun llama_get_embeddings(ctx : LlamaContext*) : Float32*
    fun llama_set_causal_attn(ctx : LlamaContext*, causal_attn : Bool) : Void
    fun llama_set_warmup(ctx : LlamaContext*, warmup : Bool) : Void
    fun llama_set_abort_callback(ctx : LlamaContext*, abort_callback : Void*, abort_callback_data : Void*) : Void
    fun llama_get_embeddings_ith(ctx : LlamaContext*, i : Int32) : Float32*
    fun llama_get_embeddings_seq(ctx : LlamaContext*, seq_id : LlamaSeqId) : Float32*
    fun llama_pooling_type(ctx : LlamaContext*) : LlamaPoolingType
    fun llama_synchronize(ctx : LlamaContext*) : Void

    # Sampler Functions
    fun llama_sampler_chain_default_params : LlamaSamplerChainParams
    fun llama_sampler_chain_init(params : LlamaSamplerChainParams) : LlamaSampler*
    fun llama_sampler_init(iface : Void*, ctx : Void*) : LlamaSampler*
    fun llama_sampler_chain_add(chain : LlamaSampler*, smpl : LlamaSampler*) : Void
    fun llama_sampler_chain_get(chain : LlamaSampler*, i : Int32) : LlamaSampler*
    fun llama_sampler_chain_n(chain : LlamaSampler*) : Int32
    fun llama_sampler_chain_remove(chain : LlamaSampler*, i : Int32) : LlamaSampler*
    fun llama_sampler_free(chain : LlamaSampler*) : Void
    fun llama_sampler_init_greedy : LlamaSampler*
    fun llama_sampler_init_top_k(k : Int32) : LlamaSampler*
    fun llama_sampler_init_top_p(p : Float32, min_keep : LibC::SizeT) : LlamaSampler*
    fun llama_sampler_init_temp(t : Float32) : LlamaSampler*
    fun llama_sampler_init_dist(seed : UInt32) : LlamaSampler*
    fun llama_sampler_sample(smpl : LlamaSampler*, ctx : LlamaContext*, idx : Int32) : LlamaToken
    fun llama_sampler_accept(smpl : LlamaSampler*, token : LlamaToken) : Void
    fun llama_sampler_reset(smpl : LlamaSampler*) : Void
    fun llama_sampler_clone(smpl : LlamaSampler*) : LlamaSampler*
    fun llama_sampler_init_min_p(p : Float32, min_keep : LibC::SizeT) : LlamaSampler*
    fun llama_sampler_init_typical(p : Float32, min_keep : LibC::SizeT) : LlamaSampler*
    fun llama_sampler_init_temp_ext(t : Float32, delta : Float32, exponent : Float32) : LlamaSampler*
    fun llama_sampler_init_top_n_sigma(n : Float32) : LlamaSampler*
    fun llama_sampler_init_xtc(p : Float32, t : Float32, min_keep : LibC::SizeT, seed : UInt32) : LlamaSampler*
    fun llama_sampler_init_infill(vocab : LlamaVocab*) : LlamaSampler*
    fun llama_sampler_init_mirostat(n_vocab : Int32, seed : UInt32, tau : Float32, eta : Float32, m : Int32) : LlamaSampler*
    fun llama_sampler_init_mirostat_v2(seed : UInt32, tau : Float32, eta : Float32) : LlamaSampler*
    fun llama_sampler_init_grammar(vocab : LlamaVocab*, grammar_str : LibC::Char*, grammar_root : LibC::Char*) : LlamaSampler*
    fun llama_sampler_init_grammar_lazy_patterns(
      vocab : LlamaVocab*,
      grammar_str : LibC::Char*,
      grammar_root : LibC::Char*,
      trigger_patterns : LibC::Char**,
      num_trigger_patterns : LibC::SizeT,
      trigger_tokens : LlamaToken*,
      num_trigger_tokens : LibC::SizeT,
    ) : LlamaSampler*
    fun llama_sampler_init_penalties(penalty_last_n : Int32, penalty_repeat : Float32, penalty_freq : Float32, penalty_present : Float32) : LlamaSampler*
    fun llama_sampler_init_dry(vocab : LlamaVocab*, n_ctx_train : Int32, dry_multiplier : Float32, dry_base : Float32, dry_allowed_length : Int32, dry_penalty_last_n : Int32, seq_breakers : LibC::Char**, num_breakers : LibC::SizeT) : LlamaSampler*
    fun llama_sampler_init_logit_bias(n_vocab : Int32, n_logit_bias : Int32, logit_bias : LlamaLogitBias*) : LlamaSampler*
    fun llama_sampler_get_seed(smpl : LlamaSampler*) : UInt32
    fun llama_sampler_name(smpl : LlamaSampler*) : LibC::Char*
    fun llama_sampler_apply(smpl : LlamaSampler*, cur_p : LlamaTokenDataArray*) : Void

    # Vocab Functions
    fun llama_tokenize(vocab : LlamaVocab*, text : LibC::Char*, text_len : Int32, tokens : LlamaToken*, n_tokens_max : Int32, add_special : Bool, parse_special : Bool) : Int32
    fun llama_token_to_piece(vocab : LlamaVocab*, token : LlamaToken, buf : LibC::Char*, length : Int32, lstrip : Int32, special : Bool) : Int32
    fun llama_detokenize(vocab : LlamaVocab*, tokens : LlamaToken*, n_tokens : Int32, text : LibC::Char*, text_len_max : Int32, remove_special : Bool, unparse_special : Bool) : Int32
    fun llama_vocab_get_text(vocab : LlamaVocab*, token : LlamaToken) : LibC::Char*
    fun llama_vocab_get_score(vocab : LlamaVocab*, token : LlamaToken) : Float32
    fun llama_vocab_get_attr(vocab : LlamaVocab*, token : LlamaToken) : LlamaTokenAttr
    fun llama_vocab_n_tokens(vocab : LlamaVocab*) : Int32
    fun llama_vocab_type(vocab : LlamaVocab*) : LlamaVocabType
    fun llama_vocab_is_eog(vocab : LlamaVocab*, token : LlamaToken) : Bool
    fun llama_vocab_is_control(vocab : LlamaVocab*, token : LlamaToken) : Bool
    fun llama_vocab_bos(vocab : LlamaVocab*) : LlamaToken
    fun llama_vocab_eos(vocab : LlamaVocab*) : LlamaToken
    fun llama_vocab_eot(vocab : LlamaVocab*) : LlamaToken
    fun llama_vocab_sep(vocab : LlamaVocab*) : LlamaToken
    fun llama_vocab_nl(vocab : LlamaVocab*) : LlamaToken
    fun llama_vocab_pad(vocab : LlamaVocab*) : LlamaToken
    fun llama_vocab_get_add_bos(vocab : LlamaVocab*) : Bool
    fun llama_vocab_get_add_eos(vocab : LlamaVocab*) : Bool
    fun llama_vocab_get_add_sep(vocab : LlamaVocab*) : Bool
    fun llama_vocab_fim_pre(vocab : LlamaVocab*) : LlamaToken
    fun llama_vocab_fim_suf(vocab : LlamaVocab*) : LlamaToken
    fun llama_vocab_fim_mid(vocab : LlamaVocab*) : LlamaToken
    fun llama_vocab_fim_pad(vocab : LlamaVocab*) : LlamaToken
    fun llama_vocab_fim_rep(vocab : LlamaVocab*) : LlamaToken
    fun llama_vocab_fim_sep(vocab : LlamaVocab*) : LlamaToken

    # Model Quantization Functions
    fun llama_model_quantize(fname_inp : LibC::Char*, fname_out : LibC::Char*, params : LlamaModelQuantizeParams*) : UInt32

    # Utility Functions
    fun llama_time_us : Int64
    fun llama_print_system_info : LibC::Char*
    # Define the callback type for logging
    alias GgmlLogCallback = Proc(Int32, LibC::Char*, Void*, Void)
    fun llama_log_set(log_callback : GgmlLogCallback, user_data : Void*) : Void
    fun llama_batch_get_one(tokens : LlamaToken*, n_tokens : Int32) : LlamaBatch
    fun llama_batch_init(n_tokens : Int32, embd : Int32, n_seq_max : Int32) : LlamaBatch
    fun llama_batch_free(batch : LlamaBatch) : Void

    # Split path functions
    fun llama_split_path(split_path : LibC::Char*, maxlen : LibC::SizeT, path_prefix : LibC::Char*, split_no : Int32, split_count : Int32) : Int32
    fun llama_split_prefix(split_prefix : LibC::Char*, maxlen : LibC::SizeT, split_path : LibC::Char*, split_no : Int32, split_count : Int32) : Int32

    # Performance utils
    struct LlamaPerfContextData
      t_start_ms : Float64
      t_load_ms : Float64
      t_p_eval_ms : Float64
      t_eval_ms : Float64
      n_p_eval : Int32
      n_eval : Int32
    end

    struct LlamaPerfSamplerData
      t_sample_ms : Float64
      n_sample : Int32
    end

    fun llama_perf_context(ctx : LlamaContext*) : LlamaPerfContextData
    fun llama_perf_context_print(ctx : LlamaContext*) : Void
    fun llama_perf_context_reset(ctx : LlamaContext*) : Void
    fun llama_perf_sampler(chain : LlamaSampler*) : LlamaPerfSamplerData
    fun llama_perf_sampler_print(chain : LlamaSampler*) : Void
    fun llama_perf_sampler_reset(chain : LlamaSampler*) : Void
  end
end
