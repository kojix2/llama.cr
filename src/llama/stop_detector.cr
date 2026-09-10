module Llama
  # Buffers possible stop prefixes and emits only text known to be safe.
  class StopDetector
    getter matched : String?

    def initialize(stops : Array(String), @include_stop : Bool = false)
      raise ArgumentError.new("stop sequences must not be empty") if stops.any?(&.empty?)
      @stops = stops.dup
      @pending = ""
      @matched = nil
    end

    def push(text : String) : String
      return "" if @matched
      data = @pending + text
      if match = earliest_match(data)
        index, stop = match
        @matched = stop
        @pending = ""
        return data.byte_slice(0, index) + (@include_stop ? stop : "")
      end

      retained = longest_prefix_suffix(data)
      safe_size = data.bytesize - retained
      @pending = retained == 0 ? "" : data.byte_slice(safe_size, retained)
      safe_size == 0 ? "" : data.byte_slice(0, safe_size)
    end

    def finish : String
      return "" if @matched
      text = @pending
      @pending = ""
      text
    end

    private def earliest_match(data : String) : Tuple(Int32, String)?
      matches = @stops.compact_map { |stop| data.byte_index(stop).try { |index| {index, stop} } }
      matches.min_by? { |index, stop| {index, -stop.bytesize} }
    end

    private def longest_prefix_suffix(data : String) : Int32
      max = Math.min(data.bytesize, @stops.max_of?(&.bytesize) || 0)
      max.downto(1) do |size|
        suffix = data.byte_slice(data.bytesize - size, size)
        return size if @stops.any?(&.starts_with?(suffix))
      end
      0
    end
  end
end
