module Llama
  enum OverflowPolicy
    Error
    Shift
  end

  class ModelOptions
    getter gpu_layers : Int32
    getter use_mmap : Bool
    getter use_mlock : Bool
    getter vocab_only : Bool
    getter lazy_mode : LazyMode
    getter check_tensors : Bool

    def initialize(
      @gpu_layers : Int32 = 0,
      @use_mmap : Bool = true,
      @use_mlock : Bool = false,
      @vocab_only : Bool = false,
      @lazy_mode : LazyMode = LazyMode::AUTO,
      @check_tensors : Bool = false,
    )
    end
  end

  class ContextOptions
    getter context_size : UInt32
    getter batch_size : UInt32
    getter micro_batch_size : UInt32
    getter threads : Int32?
    getter batch_threads : Int32?
    getter embeddings : Bool
    getter offload_kqv : Bool
    getter op_offload : Bool

    def initialize(
      @context_size : UInt32 = 0_u32,
      @batch_size : UInt32 = 512_u32,
      @micro_batch_size : UInt32 = 512_u32,
      @threads : Int32? = nil,
      @batch_threads : Int32? = nil,
      @embeddings : Bool = false,
      @offload_kqv : Bool = false,
      @op_offload : Bool = false,
    )
      raise ArgumentError.new("batch_size must be positive") if @batch_size == 0
      raise ArgumentError.new("micro_batch_size must be positive") if @micro_batch_size == 0
      raise ArgumentError.new("threads must be positive") if @threads.try { |v| v <= 0 }
      raise ArgumentError.new("batch_threads must be positive") if @batch_threads.try { |v| v <= 0 }
    end
  end

  class GenerationOptions
    getter max_tokens : Int32
    getter sampling : Sampling::Plan
    getter overflow : OverflowPolicy
    getter keep_tokens : Int32
    getter include_stop : Bool
    getter render_special : Bool
    getter cancellation : Cancellation?

    def initialize(
      @max_tokens : Int32 = 256,
      stop : Array(String) = [] of String,
      @sampling : Sampling::Plan = Sampling.default,
      @overflow : OverflowPolicy = OverflowPolicy::Error,
      @keep_tokens : Int32 = 0,
      @include_stop : Bool = false,
      @render_special : Bool = false,
      @cancellation : Cancellation? = nil,
    )
      raise ArgumentError.new("max_tokens must be positive") if @max_tokens <= 0
      raise ArgumentError.new("stop sequences must not be empty") if stop.any?(&.empty?)
      raise ArgumentError.new("keep_tokens must be non-negative") if @keep_tokens < 0
      @stop = stop.dup
    end

    def stop : Array(String)
      @stop.dup
    end

    @stop : Array(String)
  end
end
