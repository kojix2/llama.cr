require "./unit_helper"

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

  it "preserves representative text at every byte split" do
    ["ASCII", "日本語", "😀", "e\u0301"].each do |text|
      bytes = text.to_slice
      (0..bytes.size).each do |split|
        decoder = Llama::StreamingDecoder.new
        output = decoder.push(bytes[0, split])
        output += decoder.push(bytes[split, bytes.size - split])
        output += decoder.finish
        output.should eq(text)
        output.valid_encoding?.should be_true
      end
    end
  end

  it "rejects overlong and surrogate UTF-8 sequences" do
    decoder = Llama::StreamingDecoder.new
    (decoder.push(Bytes[0xc0, 0xaf]) + decoder.finish).should eq("��")

    decoder = Llama::StreamingDecoder.new
    (decoder.push(Bytes[0xed, 0xa0, 0x80]) + decoder.finish).should eq("���")
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

  it "finds stop strings at every chunk boundary" do
    text = "beforeENDafter"
    (0..text.bytesize).each do |split|
      detector = Llama::StopDetector.new(["END"])
      output = detector.push(text.byte_slice(0, split))
      output += detector.push(text.byte_slice(split, text.bytesize - split))
      output += detector.finish
      output.should eq("before")
      detector.matched.should eq("END")
    end
  end

  it "prefers the longest same-position stop and tolerates duplicates" do
    detector = Llama::StopDetector.new(["END", "END!", "END"])
    detector.push("okEND!tail").should eq("ok")
    detector.matched.should eq("END!")
  end
end
