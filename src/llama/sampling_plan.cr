module Llama
  module Sampling
    abstract class Stage
      abstract def build(vocab : Vocab) : Sampler::Base

      def selector? : Bool
        false
      end
    end

    class TopK < Stage
      getter k : Int32

      def initialize(@k : Int32)
        raise ArgumentError.new("k must be positive") if @k <= 0
      end

      def build(vocab : Vocab) : Sampler::Base
        Sampler::TopK.new(k)
      end
    end

    class MinP < Stage
      getter p : Float32
      getter min_keep : Int32

      def initialize(@p : Float32, @min_keep : Int32 = 1)
        raise ArgumentError.new("p must be between 0 and 1") unless 0.0 <= @p <= 1.0
        raise ArgumentError.new("min_keep must be positive") if @min_keep == 0
      end

      def build(vocab : Vocab) : Sampler::Base
        Sampler::MinP.new(p, min_keep)
      end
    end

    class Temperature < Stage
      getter value : Float32

      def initialize(@value : Float32)
        raise ArgumentError.new("temperature must be non-negative") if @value < 0
      end

      def build(vocab : Vocab) : Sampler::Base
        Sampler::Temp.new(value)
      end
    end

    class Distribution < Stage
      getter seed : UInt32

      def initialize(@seed : UInt32 = DEFAULT_SEED)
      end

      def build(vocab : Vocab) : Sampler::Base
        Sampler::Dist.new(seed)
      end

      def selector? : Bool
        true
      end
    end

    class Greedy < Stage
      def build(vocab : Vocab) : Sampler::Base
        Sampler::Greedy.new
      end

      def selector? : Bool
        true
      end
    end

    class Plan
      def initialize(stages : Array(Stage))
        raise ArgumentError.new("sampling plan cannot be empty") if stages.empty?
        raise ArgumentError.new("sampling plan must end with a selector") unless stages.last.selector?
        if stages[0...-1].any?(&.selector?)
          raise ArgumentError.new("selector must be the final sampling stage")
        end
        @stages = stages.dup
      end

      def stages : Array(Stage)
        @stages.dup
      end

      def build(vocab : Vocab) : SamplerChain
        chain = SamplerChain.new
        @stages.each { |stage| chain.add(stage.build(vocab)) }
        chain
      rescue ex
        chain.try(&.close)
        raise ex
      end

      @stages : Array(Stage)
    end

    def self.greedy : Plan
      Plan.new([Greedy.new] of Stage)
    end

    def self.temperature(value : Float32, seed : UInt32 = DEFAULT_SEED) : Plan
      return greedy if value == 0
      Plan.new([Temperature.new(value), Distribution.new(seed)] of Stage)
    end

    def self.default(seed : UInt32 = DEFAULT_SEED) : Plan
      Plan.new([TopK.new(40), MinP.new(0.05_f32), Temperature.new(0.8_f32), Distribution.new(seed)] of Stage)
    end
  end
end
