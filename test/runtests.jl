include("macros-tests.jl")
include("linalg-tests.jl")
using .TestingMayOptimizeLinearAlgebraMethods
ldiv_tests()
cholesky_tests()

nothing
