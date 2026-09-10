module Llama
  record RuntimeInfo,
    wrapper_version : String,
    expected_build : String,
    reported_version : String,
    backend_count : UInt64,
    system_info : String,
    mmap_supported : Bool,
    mlock_supported : Bool,
    rpc_supported : Bool,
    gpu_offload_supported : Bool
end
