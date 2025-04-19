module Llama
  module Sampler
  end
end

require "./sampler/base"
require "./sampler/chain"
require "./sampler/greedy"
require "./sampler/top_k"
require "./sampler/top_p"
require "./sampler/temp"
require "./sampler/dist"
require "./sampler/min_p"
require "./sampler/typical"
require "./sampler/mirostat"
require "./sampler/mirostat_v2"
require "./sampler/grammar"
require "./sampler/penalties"
require "./sampler/temp_ext"
require "./sampler/top_n_sigma"
require "./sampler/xtc"
require "./sampler/infill"
require "./sampler/grammar_lazy_patterns"

# Backward compatibility aliases
module Llama
  # Old class names are aliased to the new module structure
  alias GreedySampler = Sampler::Greedy
  alias TopKSampler = Sampler::TopK
  alias TopPSampler = Sampler::TopP
  alias TempSampler = Sampler::Temp
  alias DistSampler = Sampler::Dist
  alias SamplerChain = Sampler::Chain
  alias MinPSampler = Sampler::MinP
  alias TypicalSampler = Sampler::Typical
  alias MirostatSampler = Sampler::Mirostat
  alias MirostatV2Sampler = Sampler::MirostatV2
  alias GrammarSampler = Sampler::Grammar
  alias PenaltiesSampler = Sampler::Penalties
  alias TempExtSampler = Sampler::TempExt
  alias TopNSigmaSampler = Sampler::TopNSigma
  alias XtcSampler = Sampler::Xtc
  alias InfillSampler = Sampler::Infill
  alias GrammarLazyPatternsSampler = Sampler::GrammarLazyPatterns
end
