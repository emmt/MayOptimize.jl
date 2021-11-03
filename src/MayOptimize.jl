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
    CholeskyBanachiewiczLower,
    CholeskyBanachiewiczLowerI,
    CholeskyBanachiewiczLowerII,
    CholeskyBanachiewiczUpper,
    CholeskyCroutLower,
    CholeskyCroutUpper,
    CholeskyCroutUpperI,
    CholeskyCroutUpperII,
    CholeskyFactorization,
    exec!

end # module
