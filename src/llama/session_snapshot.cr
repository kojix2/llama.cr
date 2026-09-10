require "json"

module Llama
  # Portable canonical transcript state for Session.
  #
  # Native KV bytes are deliberately excluded: Session currently rebuilds the
  # checked native state from this text, making snapshots portable across
  # processes while still rejecting a different model or llama.cpp release.
  class SessionSnapshot
    include JSON::Serializable

    FORMAT_VERSION       = 1
    MAX_SERIALIZED_BYTES = 64 * 1024 * 1024

    getter format_version : Int32
    getter wrapper_version : String
    getter llama_cpp_version : String
    getter model_fingerprint : String
    getter transcript : String

    def initialize(
      @transcript : String,
      @model_fingerprint : String,
      @llama_cpp_version : String = Llama.llama_cpp_version,
      @wrapper_version : String = Llama::VERSION,
      @format_version : Int32 = FORMAT_VERSION,
    )
    end
  end
end
