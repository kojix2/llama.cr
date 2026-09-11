require "./spec_helper"

CHAT_TEMPLATE = <<-'TEMPLATE'
{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}
TEMPLATE

describe Llama::Chat do
  it "closes the block factory after a caller exception" do
    model = Llama::Model.new(MODEL_PATH)
    captured = nil.as(Llama::Chat?)
    expect_raises(Exception, "boom") do
      model.chat(template: CHAT_TEMPLATE) do |chat|
        captured = chat
        raise "boom"
      end
    end
    captured.not_nil!.closed?.should be_true
    model.close
  end

  it "commits successful turns and returns defensive history" do
    model = Llama::Model.new(MODEL_PATH)
    chat = model.chat(system: "brief", template: CHAT_TEMPLATE)
    plan = Llama::Sampling::Plan.new([
      Llama::Sampling::Grammar.new(%(root ::= "ok")),
      Llama::Sampling::Greedy.new,
    ] of Llama::Sampling::Stage)

    result = chat.ask("hello", Llama::GenerationOptions.new(max_tokens: 4, sampling: plan))
    result.text.should eq("ok")
    chat.history.map(&.role).should eq(["system", "user", "assistant"])
    copy = chat.history
    copy.clear
    chat.history.size.should eq(3)

    chat.close
    model.close
  end

  it "does not commit cancelled turns by default" do
    model = Llama::Model.new(MODEL_PATH)
    chat = model.chat(template: CHAT_TEMPLATE)
    cancellation = Llama::Cancellation.new
    cancellation.cancel

    result = chat.ask("hello", Llama::GenerationOptions.new(cancellation: cancellation))
    result.finish_reason.should eq(Llama::FinishReason::Cancelled)
    chat.history.should be_empty

    chat.close
    model.close
  end

  it "rejects history mutation from inside an active turn" do
    model = Llama::Model.new(MODEL_PATH)
    chat = model.chat(template: CHAT_TEMPLATE)
    plan = Llama::Sampling::Plan.new([
      Llama::Sampling::Grammar.new(%(root ::= "ok")),
      Llama::Sampling::Greedy.new,
    ] of Llama::Sampling::Stage)

    chat.ask("hello", Llama::GenerationOptions.new(max_tokens: 4, sampling: plan)) do |_chunk|
      expect_raises(Llama::BusyError, "Llama::Chat is busy") { chat.ask("again") }
      expect_raises(Llama::BusyError, "Llama::Chat is busy") { chat.clear }
      expect_raises(Llama::BusyError, "Llama::Chat is busy") { chat.close }
      expect_raises(Llama::BusyError, "Llama::Model is busy") { model.close }
    end

    chat.history.size.should eq(2)
    chat.clear
    chat.history.should be_empty
    chat.close
    model.close
  end
end
