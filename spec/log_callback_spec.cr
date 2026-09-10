require "./spec_helper"

describe "Llama log callback" do
  it "stays on the caller thread in b10809 and contains callback exceptions" do
    callback_threads = [] of UInt64
    mutex = Mutex.new
    caller_thread = Thread.current.object_id
    should_raise = true

    Llama.log_set do |_level, _message|
      mutex.synchronize { callback_threads << Thread.current.object_id }
      if should_raise
        should_raise = false
        raise "log callback failure"
      end
    end

    model = Llama::Model.new(MODEL_PATH)
    context = model.context(n_threads: 2, n_threads_batch: 2)
    context.generate("hello", 2, 0.0)

    callback_threads.should_not be_empty
    callback_threads.uniq.should eq([caller_thread])
    Llama.take_log_callback_error.try(&.message).should eq("log callback failure")
    Llama.take_log_callback_error.should be_nil
  ensure
    context.try(&.close)
    model.try(&.close)
    Llama.log_level = Llama::LOG_LEVEL_ERROR
  end
end
