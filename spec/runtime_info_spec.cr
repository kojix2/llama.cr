require "./spec_helper"

describe Llama::RuntimeInfo do
  it "reports the pinned runtime and loaded backend capabilities" do
    Llama.check_compatibility!.should be_nil
    Llama.llama_cpp_version.should eq(Llama::LLAMA_CPP_REPORTED_VERSION)

    info = Llama.runtime_info
    info.wrapper_version.should eq(Llama::VERSION)
    info.expected_build.should eq("b10809")
    info.reported_version.should eq("0.4.0")
    info.backend_count.should be > 0_u64
    info.system_info.should_not be_empty
  end
end
