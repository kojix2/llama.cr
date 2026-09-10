module Llama
  # Thread-safe cooperative cancellation shared with high-level operations.
  class Cancellation
    def initialize
      @cancelled = Atomic(Bool).new(false)
    end

    def cancel : Nil
      @cancelled.set(true)
    end

    def cancelled? : Bool
      @cancelled.get
    end

    @cancelled : Atomic(Bool)
  end
end
