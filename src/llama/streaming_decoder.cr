module Llama
  # Converts token byte pieces into chunks that are always valid UTF-8.
  class StreamingDecoder
    def initialize(@tokenizer : Tokenizer)
      @pending = [] of UInt8
    end

    def push(token : Token) : String
      push(@tokenizer.piece(token))
    end

    # Byte-oriented entry point useful for adapters and deterministic tests.
    def push(bytes : Bytes) : String
      @pending.concat(bytes)
      consume(final: false)
    end

    def finish : String
      consume(final: true)
    end

    def reset : Nil
      @pending.clear
    end

    private def consume(final : Bool) : String
      output = IO::Memory.new
      i = 0
      while i < @pending.size
        first = @pending[i]
        width = utf8_width(first)
        if width == 0
          output << '\uFFFD'
          i += 1
          next
        end
        if i + width > @pending.size
          break unless final
          output << '\uFFFD'
          i = @pending.size
          break
        end
        unless valid_sequence?(i, width)
          output << '\uFFFD'
          i += 1
          next
        end
        width.times { |offset| output.write_byte(@pending[i + offset]) }
        i += width
      end
      @pending = i < @pending.size ? @pending[i..] : [] of UInt8
      output.to_s
    end

    private def utf8_width(byte : UInt8) : Int32
      return 1 if byte <= 0x7f
      return 2 if 0xc2 <= byte <= 0xdf
      return 3 if 0xe0 <= byte <= 0xef
      return 4 if 0xf0 <= byte <= 0xf4
      0
    end

    private def valid_sequence?(index : Int32, width : Int32) : Bool
      return true if width == 1
      continuation = ->(b : UInt8) { 0x80 <= b <= 0xbf }
      return false unless (1...width).all? { |offset| continuation.call(@pending[index + offset]) }
      first = @pending[index]
      second = @pending[index + 1]
      return false if first == 0xe0 && second < 0xa0
      return false if first == 0xed && second > 0x9f
      return false if first == 0xf0 && second < 0x90
      return false if first == 0xf4 && second > 0x8f
      true
    end
  end
end
