require "./spec_helper"

describe Llama::StreamingDecoder do
  it "buffers split UTF-8 sequences" do
    decoder = Llama::StreamingDecoder.new

    decoder.push(Bytes[0xe3, 0x81]).should eq("")
    decoder.push(Bytes[0x82, 0xf0, 0x9f]).should eq("あ")
    decoder.push(Bytes[0x98, 0x80]).should eq("😀")
    decoder.finish.should eq("")
  end

  it "replaces invalid and incomplete terminal bytes" do
    decoder = Llama::StreamingDecoder.new
    decoder.push(Bytes[0xff]).should eq("�")
    decoder.push(Bytes[0xe3]).should eq("")
    decoder.finish.should eq("�")
  end
end

describe Llama::StopDetector do
  it "detects stops spanning chunks without emitting them" do
    detector = Llama::StopDetector.new(["END", "END!"])
    detector.push("hello E").should eq("hello ")
    detector.push("ND!ignored").should eq("")
    detector.matched.should eq("END!")
    detector.finish.should eq("")
  end

  it "flushes unmatched prefixes" do
    detector = Llama::StopDetector.new(["stop"])
    detector.push("text st").should eq("text ")
    detector.finish.should eq("st")
  end

  it "can include matched stop text" do
    detector = Llama::StopDetector.new(["END"], include_stop: true)
    detector.push("okENDafter").should eq("okEND")
  end
end
