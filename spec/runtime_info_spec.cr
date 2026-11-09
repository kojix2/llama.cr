require "./spec_helper"

describe Llama::RuntimeInfo do
  it "reports the loaded runtime and backend capabilities" do
    info = Llama.runtime_info
    info.wrapper_version.should eq(Llama::VERSION)
    info.expected_build.should eq("b12110")
    info.reported_version.should eq(Llama.llama_cpp_version)
    info.reported_version.should_not be_empty
    Llama.check_compatibility!
    info.backend_count.should be > 0_u64
    info.system_info.should_not be_empty
  end
end
