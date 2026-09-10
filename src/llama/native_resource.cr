module Llama
  # Shared state checks for wrappers that own native resources.
  module NativeResource
    private def ensure_open! : Nil
      raise ClosedError.new(self.class.to_s) if closed?
    end
  end
end
