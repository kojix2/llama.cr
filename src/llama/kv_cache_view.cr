require "./kv_cache_view/error"

module Llama
  # DEPRECATED: KV Cache View functionality has been removed from llama.cpp
  # This class is temporarily disabled until migration to new memory APIs
  # See: https://github.com/ggml-org/llama.cpp/pull/XXXX

  # TODO: Implement replacement using new memory management APIs
  # The KV cache view functionality needs to be reimplemented using
  # the new llama_memory_* functions instead of the removed llama_kv_cache_view_* functions

  # Placeholder classes to maintain API compatibility
  class KvCacheViewCell
    def initialize(@pos : Int32 = -1)
    end

    def pos : Int32
      @pos
    end

    def to_s(io : IO) : Nil
      io << "KvCacheViewCell(pos: #{pos}) [DEPRECATED]"
    end

    def inspect(io : IO) : Nil
      to_s(io)
    end
  end

  class KvCacheView
    def initialize(ctx : Context, n_seq_max : Int32 = 4)
      raise Error.new("KvCacheView is deprecated and temporarily disabled. Use the new memory management APIs instead.")
    end

    def initialize(ctx : Context, n_seq_max : Int32 = 4, &block)
      raise Error.new("KvCacheView is deprecated and temporarily disabled. Use the new memory management APIs instead.")
    end
  end
end
