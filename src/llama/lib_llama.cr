module Llama
  @[Link("llama")]
  lib LibLlama
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
    type LlamaSampler = Void*
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
      ALL_F32       =  0
      MOSTLY_F16    =  1
      MOSTLY_Q4_0   =  2
      MOSTLY_Q4_1   =  3
      MOSTLY_Q8_0   =  7
      MOSTLY_Q5_0   =  8
      MOSTLY_Q5_1   =  9
      MOSTLY_Q2_K   = 10
      MOSTLY_Q3_K_S = 11
      MOSTLY_Q3_K_M = 12
      MOSTLY_Q3_K_L = 13
      MOSTLY_Q4_K_S = 14
      MOSTLY_Q4_K_M = 15
      MOSTLY_Q5_K_S = 16
      MOSTLY_Q5_K_M = 17
      MOSTLY_Q6_K   = 18
      # Add other values as needed

      GUESSED = 1024 # Not specified in the model file
    end

    enum LlamaRopeType
      NONE   = -1
      NORM   =  0
      NEOX   =  1
      MROPE  =  2
      VISION =  3
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

    struct LlamaBatch
      n_tokens : Int32
      token : LlamaToken*
      embd : Float32*
      pos : LlamaPos*
      n_seq_id : Int32*
      seq_id : LlamaSeqId**
      logits : Int8*
    end

    # Model parameters
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

    # Context related functions
    fun llama_init_from_model(model : LlamaModel*, params : LlamaContextParams) : LlamaContext*
    fun llama_free(ctx : LlamaContext*)

    # Inference related functions
    fun llama_decode(ctx : LlamaContext*, batch : LlamaBatch) : Int32

    # Tokenization related functions
    fun llama_tokenize(vocab : LlamaVocab*, text : LibC::Char*, text_len : Int32, tokens : LlamaToken*, n_tokens_max : Int32, add_special : Bool, parse_special : Bool) : Int32

    # Vocabulary related functions
    fun llama_model_get_vocab(model : LlamaModel*) : LlamaVocab*
    fun llama_vocab_get_text(vocab : LlamaVocab*, token : LlamaToken) : LibC::Char*

    # System information
    fun llama_print_system_info : LibC::Char*

    # Note: This is a partial implementation. More functions and structures will be added as needed.
  end
end
