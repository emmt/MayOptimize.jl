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
include("linalg.jl")
import .LinearAlgebraMethods:
    CholeskyLowerColumnwise,
    CholeskyLowerRowwiseI,
    CholeskyLowerRowwiseII,
    CholeskyUpperColumnwiseI,
    CholeskyUpperColumnwiseII,
    CholeskyUpperRowwise,
    exec!

end # module
