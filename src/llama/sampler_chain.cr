module Llama
  # SamplerChain: manages a chain of samplers, but is not itself a Sampler::Base.
  # Ownership and lifecycle are managed internally.
  class SamplerChain
    # Creates a new SamplerChain with optional parameters
    #
    # Parameters:
    # - no_perf: Whether to disable performance counters (default: false)
    #
    # Raises:
    # - Llama::Error if the sampler chain cannot be created
    def initialize(no_perf : Bool = false)
      params = LibLlama.llama_sampler_chain_default_params
      params.no_perf = no_perf
      @handle = LibLlama.llama_sampler_chain_init(params)
      raise Error.new("Failed to create sampler chain") if @handle.null?
      @samplers = [] of Sampler::Base
    end

    # Adds a sampler to the chain
    #
    # Parameters:
    # - sampler: The sampler to add to the chain
    def add(sampler : Sampler::Base)
      LibLlama.llama_sampler_chain_add(@handle, sampler.to_unsafe)
      sampler.set_owned_by_chain(true)
      # Ownership is transferred to the C side; do not free sampler separately.
      @samplers << sampler # Keep reference to prevent GC
    end

    # Removes a sampler from the chain at the given index.
    # Returns the original Sampler::Base instance with ownership restored.
    def remove(index : Int)
      # Remove from C chain and get the C pointer
      removed_ptr = LibLlama.llama_sampler_chain_remove(@handle, index)
      raise Error.new("Failed to remove sampler at index #{index}") if removed_ptr.null?
      # Find the matching Sampler::Base instance in @samplers
      sampler = @samplers.delete_at(index)
      # Update the handle in case the C API returned a different pointer
      sampler.set_handle(removed_ptr)
      # Restore ownership to the caller
      sampler.set_owned_by_chain(false)
      sampler
    end

    # Samples a token using the sampler chain
    #
    # Parameters:
    # - ctx: The context to sample from
    # - idx: The index of the logits to sample from (-1 for the last token)
    #
    # Returns:
    # - The sampled token
    def sample(ctx : Context, idx : Int32 = -1) : Int32
      LibLlama.llama_sampler_sample(@handle, ctx.to_unsafe, idx)
    end

    # Accepts a token, updating the internal state of the samplers
    #
    # Parameters:
    # - token: The token to accept
    def accept(token : Int32)
      LibLlama.llama_sampler_accept(@handle, token)
    end

    # Frees the resources associated with this sampler chain
    def finalize
      @samplers.clear if @samplers
      LibLlama.llama_sampler_free(@handle) if @handle && !@handle.null?
    end

    # Print performance information for this sampler chain
    def print_perf
      LibLlama.llama_perf_sampler_print(@handle)
    end

    # Reset performance counters for this sampler chain
    def reset_perf
      LibLlama.llama_perf_sampler_reset(@handle)
    end

    # For C API compatibility
    def to_unsafe
      @handle
    end

    @handle : LibLlama::LlamaSampler*
    @samplers : Array(Sampler::Base)

    # :nodoc:
    def clone
      raise NotImplementedError.new("clone is not supported for #{self.class}")
    end

    # :nodoc:
    def dup
      raise NotImplementedError.new("dup is not supported for #{self.class}")
    end
  end
end
