module MayOptimize

export
    Debug,
    InBounds,
    OptimLevel,
    Vectorize,
    @maybe_inbounds,
    @maybe_vectorized

include("types.jl")
include("macros.jl")

end # module
