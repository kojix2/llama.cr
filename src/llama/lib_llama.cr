module Llama
  @[Link("llama")]
  lib LibLlama
    # Note: The following functions marked as DEPRECATED in llama.cpp have been removed
    # - llama_load_model_from_file → replaced with llama_model_load_from_file
    # - llama_free_model → replaced with llama_model_free
    # - llama_new_context_with_model → replaced with llama_init_from_model
    # - llama_n_ctx_train, llama_n_embd, llama_n_layer, llama_n_head → replaced with llama_model_* functions
    # - llama_n_vocab → replaced with llama_vocab_n_tokens
    # - llama_token_* functions → replaced with llama_vocab_* functions
    # - llama_kv_cache_* functions → replaced with llama_kv_self_* functions
    # - llama_get_state_size, llama_copy_state_data, llama_set_state_data → replaced with llama_state_* functions
    # - llama_load_session_file, llama_save_session_file → replaced with llama_state_load_file, llama_state_save_file

    # Constants
    LLAMA_DEFAULT_SEED = 0xFFFFFFFF_u32
    LLAMA_TOKEN_NULL   =             -1

    # Basic types
    alias LlamaToken = Int32
    alias LlamaPos = Int32
    alias LlamaSeqId = Int32

    # Forward declarations of opaque structures
    type LlamaVocab = Void*
    type LlamaModel = Void*
    type LlamaContext = Void*
    type LlamaKvCache = Void*

    # Enumerations
    enum LlamaVocabType
      NONE = 0 # For models without vocab
      SPM  = 1 # LLaMA tokenizer based on byte-level BPE with byte fallback
      BPE  = 2 # GPT-2 tokenizer based on byte-level BPE
      WPM  = 3 # BERT tokenizer based on WordPiece
      UGM  = 4 # T5 tokenizer based on Unigram
      RWKV = 5 # RWKV tokenizer based on greedy tokenization
    end

    enum LlamaFtype
      ALL_F32     = 0
      MOSTLY_F16  = 1 # except 1d tensors
      MOSTLY_Q4_0 = 2 # except 1d tensors
      MOSTLY_Q4_1 = 3 # except 1d tensors
      # MOSTLY_Q4_1_SOME_F16 = 4  # tok_embeddings.weight and output.weight are F16
      # MOSTLY_Q4_2   = 5  # support has been removed
      # MOSTLY_Q4_3   = 6  # support has been removed
      MOSTLY_Q8_0    =  7 # except 1d tensors
      MOSTLY_Q5_0    =  8 # except 1d tensors
      MOSTLY_Q5_1    =  9 # except 1d tensors
      MOSTLY_Q2_K    = 10 # except 1d tensors
      MOSTLY_Q3_K_S  = 11 # except 1d tensors
      MOSTLY_Q3_K_M  = 12 # except 1d tensors
      MOSTLY_Q3_K_L  = 13 # except 1d tensors
      MOSTLY_Q4_K_S  = 14 # except 1d tensors
      MOSTLY_Q4_K_M  = 15 # except 1d tensors
      MOSTLY_Q5_K_S  = 16 # except 1d tensors
      MOSTLY_Q5_K_M  = 17 # except 1d tensors
      MOSTLY_Q6_K    = 18 # except 1d tensors
      MOSTLY_IQ2_XXS = 19 # except 1d tensors
      MOSTLY_IQ2_XS  = 20 # except 1d tensors
      MOSTLY_Q2_K_S  = 21 # except 1d tensors
      MOSTLY_IQ3_XS  = 22 # except 1d tensors
      MOSTLY_IQ3_XXS = 23 # except 1d tensors
      MOSTLY_IQ1_S   = 24 # except 1d tensors
      MOSTLY_IQ4_NL  = 25 # except 1d tensors
      MOSTLY_IQ3_S   = 26 # except 1d tensors
      MOSTLY_IQ3_M   = 27 # except 1d tensors
      MOSTLY_IQ2_S   = 28 # except 1d tensors
      MOSTLY_IQ2_M   = 29 # except 1d tensors
      MOSTLY_IQ4_XS  = 30 # except 1d tensors
      MOSTLY_IQ1_M   = 31 # except 1d tensors
      MOSTLY_BF16    = 32 # except 1d tensors
      # MOSTLY_Q4_0_4_4 = 33  # removed from gguf files, use Q4_0 and runtime repack
      # MOSTLY_Q4_0_4_8 = 34  # removed from gguf files, use Q4_0 and runtime repack
      # MOSTLY_Q4_0_8_8 = 35  # removed from gguf files, use Q4_0 and runtime repack
      MOSTLY_TQ1_0 = 36 # except 1d tensors
      MOSTLY_TQ2_0 = 37 # except 1d tensors

      GUESSED = 1024 # not specified in the model file
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
    end

    enum LlamaAttentionType
      UNSPECIFIED = -1
      CAUSAL      =  0
      NON_CAUSAL  =  1
    end

    enum LlamaSplitMode
      NONE  = 0 # single GPU
      LAYER = 1 # split layers and KV across GPUs
      ROW   = 2 # split layers and KV across GPUs, use tensor parallelism if supported
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
      # Union for values - Crystal doesn't support C unions directly
      # so we include all possible fields
      val_i64 : Int64
      val_f64 : Float64
      val_bool : Bool
      val_str : LibC::Char[128]
    end

    # Basic structures
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

    # Batch structure for processing tokens
    struct LlamaBatch
      n_tokens : Int32
      token : LlamaToken*
      embd : Float32*
      pos : LlamaPos*
      n_seq_id : Int32*
      seq_id : LlamaSeqId**
      logits : Int8*
    end

    # Chat message structure
    struct LlamaChatMessage
      role : LibC::Char*
      content : LibC::Char*
    end

    # Chat template functions
    fun llama_chat_apply_template(
      tmpl : LibC::Char*,
      chat : LlamaChatMessage*,
      n_msg : LibC::SizeT,
      add_ass : Bool,
      buf : LibC::Char*,
      length : Int32,
    ) : Int32

    # Get chat template from model
    fun llama_model_chat_template(model : LlamaModel*, name : LibC::Char*) : LibC::Char*

    # Get list of built-in chat templates
    fun llama_chat_builtin_templates(output : LibC::Char**, len : LibC::SizeT) : Int32

    type LlamaSampler = Void*

    # Sampler chain parameters
    struct LlamaSamplerChainParams
      no_perf : Bool
    end

    # Sampler-related functions
    fun llama_sampler_chain_default_params : LlamaSamplerChainParams
    fun llama_sampler_chain_init(params : LlamaSamplerChainParams) : LlamaSampler*
    fun llama_sampler_chain_add(chain : LlamaSampler*, smpl : LlamaSampler*) : Void
    fun llama_sampler_free(smpl : LlamaSampler*) : Void

    # Sampler initialization functions
    fun llama_sampler_init_top_k(k : Int32) : LlamaSampler*
    fun llama_sampler_init_top_p(p : Float32, min_keep : LibC::SizeT) : LlamaSampler*
    fun llama_sampler_init_temp(t : Float32) : LlamaSampler*
    fun llama_sampler_init_dist(seed : UInt32) : LlamaSampler*

    # Sampling functions
    fun llama_sampler_sample(smpl : LlamaSampler*, ctx : LlamaContext*, idx : Int32) : LlamaToken
    fun llama_sampler_accept(smpl : LlamaSampler*, token : LlamaToken) : Void

    struct LlamaModelParams
      # NULL-terminated list of devices to use for offloading
      devices : Void*

      # NULL-terminated list of buffer types to use for tensors that match a pattern
      tensor_buft_overrides : Void*

      n_gpu_layers : Int32
      split_mode : Int32
      main_gpu : Int32

      # Proportion of the model to offload to each GPU
      tensor_split : Float32*

      # Progress callback
      progress_callback : Void*
      progress_callback_user_data : Void*

      # Override key-value pairs of the model meta data
      kv_overrides : Void*

      # Boolean flags
      vocab_only : Bool
      use_mmap : Bool
      use_mlock : Bool
      check_tensors : Bool
    end

    # Context parameters
    struct LlamaContextParams
      n_ctx : UInt32
      n_batch : UInt32
      n_ubatch : UInt32
      n_seq_max : UInt32
      n_threads : Int32
      n_threads_batch : Int32

      rope_scaling_type : Int32
      pooling_type : Int32
      attention_type : Int32

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

      # Boolean flags
      logits_all : Bool
      embeddings : Bool
      offload_kqv : Bool
      flash_attn : Bool
      no_perf : Bool

      # Abort callback
      abort_callback : Void*
      abort_callback_data : Void*
    end

    # Helper functions for parameter structures
    fun llama_model_default_params : LlamaModelParams
    fun llama_context_default_params : LlamaContextParams

    # Model related functions
    fun llama_model_load_from_file(path_model : LibC::Char*, params : LlamaModelParams) : LlamaModel*
    fun llama_model_free(model : LlamaModel*)
    fun llama_model_n_params(model : LlamaModel*) : UInt64
    fun llama_model_n_embd(model : LlamaModel*) : Int32
    fun llama_model_n_layer(model : LlamaModel*) : Int32
    fun llama_model_n_head(model : LlamaModel*) : Int32
    fun llama_model_n_head_kv(model : LlamaModel*) : Int32
    fun llama_model_has_encoder(model : LlamaModel*) : Bool
    fun llama_model_has_decoder(model : LlamaModel*) : Bool
    fun llama_model_is_recurrent(model : LlamaModel*) : Bool
    fun llama_model_rope_freq_scale_train(model : LlamaModel*) : Float32
    fun llama_model_decoder_start_token(model : LlamaModel*) : LlamaToken

    # Context related functions
    fun llama_init_from_model(model : LlamaModel*, params : LlamaContextParams) : LlamaContext*
    fun llama_free(ctx : LlamaContext*)

    # Inference related functions
    fun llama_encode(ctx : LlamaContext*, batch : LlamaBatch) : Int32
    fun llama_decode(ctx : LlamaContext*, batch : LlamaBatch) : Int32
    fun llama_get_logits(ctx : LlamaContext*) : Float32*

    # Tokenization related functions
    fun llama_tokenize(vocab : LlamaVocab*, text : LibC::Char*, text_len : Int32, tokens : LlamaToken*, n_tokens_max : Int32, add_special : Bool, parse_special : Bool) : Int32

    # Vocabulary related functions
    fun llama_model_get_vocab(model : LlamaModel*) : LlamaVocab*
    fun llama_vocab_get_text(vocab : LlamaVocab*, token : LlamaToken) : LibC::Char*
    fun llama_vocab_n_tokens(vocab : LlamaVocab*) : Int32
    fun llama_vocab_bos(vocab : LlamaVocab*) : LlamaToken
    fun llama_vocab_eos(vocab : LlamaVocab*) : LlamaToken
    fun llama_vocab_eot(vocab : LlamaVocab*) : LlamaToken
    fun llama_vocab_nl(vocab : LlamaVocab*) : LlamaToken
    fun llama_vocab_pad(vocab : LlamaVocab*) : LlamaToken

    # System information
    fun llama_print_system_info : LibC::Char*

    # ===== NEW ADDITIONS =====

    # 1. KV Cache Management Functions
    fun llama_get_kv_self(ctx : LlamaContext*) : LlamaKvCache*
    fun llama_kv_self_clear(ctx : LlamaContext*) : Void
    fun llama_kv_self_n_tokens(ctx : LlamaContext*) : Int32
    fun llama_kv_self_used_cells(ctx : LlamaContext*) : Int32
    fun llama_kv_self_seq_rm(ctx : LlamaContext*, seq_id : LlamaSeqId, p0 : LlamaPos, p1 : LlamaPos) : Bool
    fun llama_kv_self_seq_cp(ctx : LlamaContext*, seq_id_src : LlamaSeqId, seq_id_dst : LlamaSeqId, p0 : LlamaPos, p1 : LlamaPos) : Void
    fun llama_kv_self_seq_keep(ctx : LlamaContext*, seq_id : LlamaSeqId) : Void
    fun llama_kv_self_seq_add(ctx : LlamaContext*, seq_id : LlamaSeqId, p0 : LlamaPos, p1 : LlamaPos, delta : LlamaPos) : Void
    fun llama_kv_self_seq_div(ctx : LlamaContext*, seq_id : LlamaSeqId, p0 : LlamaPos, p1 : LlamaPos, d : Int32) : Void
    fun llama_kv_self_seq_pos_max(ctx : LlamaContext*, seq_id : LlamaSeqId) : LlamaPos
    fun llama_kv_self_defrag(ctx : LlamaContext*) : Void
    fun llama_kv_self_can_shift(ctx : LlamaContext*) : Bool
    fun llama_kv_self_update(ctx : LlamaContext*) : Void

    # 2. Batch Processing Functions
    fun llama_batch_get_one(tokens : LlamaToken*, n_tokens : Int32) : LlamaBatch
    fun llama_batch_init(n_tokens : Int32, embd : Int32, n_seq_max : Int32) : LlamaBatch
    fun llama_batch_free(batch : LlamaBatch) : Void

    # 3. State Management Functions
    fun llama_state_get_size(ctx : LlamaContext*) : LibC::SizeT
    fun llama_state_get_data(ctx : LlamaContext*, dst : UInt8*, size : LibC::SizeT) : LibC::SizeT
    fun llama_state_set_data(ctx : LlamaContext*, src : UInt8*, size : LibC::SizeT) : LibC::SizeT
    fun llama_state_load_file(ctx : LlamaContext*, path_session : LibC::Char*, tokens_out : LlamaToken*, n_token_capacity : LibC::SizeT, n_token_count_out : LibC::SizeT*) : Bool
    fun llama_state_save_file(ctx : LlamaContext*, path_session : LibC::Char*, tokens : LlamaToken*, n_token_count : LibC::SizeT) : Bool
    fun llama_state_seq_get_size(ctx : LlamaContext*, seq_id : LlamaSeqId) : LibC::SizeT
    fun llama_state_seq_get_data(ctx : LlamaContext*, dst : UInt8*, size : LibC::SizeT, seq_id : LlamaSeqId) : LibC::SizeT
    fun llama_state_seq_set_data(ctx : LlamaContext*, src : UInt8*, size : LibC::SizeT, dest_seq_id : LlamaSeqId) : LibC::SizeT
    fun llama_state_seq_save_file(ctx : LlamaContext*, filepath : LibC::Char*, seq_id : LlamaSeqId, tokens : LlamaToken*, n_token_count : LibC::SizeT) : LibC::SizeT
    fun llama_state_seq_load_file(ctx : LlamaContext*, filepath : LibC::Char*, dest_seq_id : LlamaSeqId, tokens_out : LlamaToken*, n_token_capacity : LibC::SizeT, n_token_count_out : LibC::SizeT*) : LibC::SizeT

    # ===== EMBEDDINGS FUNCTIONS =====

    # Pooling types for embeddings
    enum LlamaPoolingType
      NONE = 0
      MEAN = 1
      CLS  = 2
      LAST = 3
      RANK = 4
    end

    # Embeddings related functions
    fun llama_set_embeddings(ctx : LlamaContext*, embeddings : Bool) : Void
    fun llama_get_embeddings(ctx : LlamaContext*) : Float32*
    fun llama_get_embeddings_ith(ctx : LlamaContext*, i : Int32) : Float32*
    fun llama_get_embeddings_seq(ctx : LlamaContext*, seq_id : LlamaSeqId) : Float32*
    fun llama_pooling_type(ctx : LlamaContext*) : LlamaPoolingType

    # ===== ADVANCED SAMPLING METHODS =====

    # Advanced sampler initialization functions
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

    # ===== MODEL METADATA FUNCTIONS =====

    # Model metadata access functions
    fun llama_model_meta_val_str(model : LlamaModel*, key : LibC::Char*, buf : LibC::Char*, buf_size : LibC::SizeT) : Int32
    fun llama_model_meta_count(model : LlamaModel*) : Int32
    fun llama_model_meta_key_by_index(model : LlamaModel*, i : Int32, buf : LibC::Char*, buf_size : LibC::SizeT) : Int32
    fun llama_model_meta_val_str_by_index(model : LlamaModel*, i : Int32, buf : LibC::Char*, buf_size : LibC::SizeT) : Int32
    fun llama_model_desc(model : LlamaModel*, buf : LibC::Char*, buf_size : LibC::SizeT) : Int32
  end
end
