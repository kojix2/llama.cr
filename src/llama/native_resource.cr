module Llama
  # Shared state checks for wrappers that own native resources.
  module NativeResource
    # Stable for the lifetime of this Crystal wrapper; used by parent registries.
    def resource_id : UInt64
      object_id
    end

    private def ensure_open! : Nil
      raise ClosedError.new(self.class.to_s) if closed?
    end
  end
end
