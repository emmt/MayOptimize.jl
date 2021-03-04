module BenchmarkingMayOptimizeMacros

using BenchmarkTools, LinearAlgebra
using MayOptimize
using MayOptimize: Basic, AVX

ops = ((:Basic, "----"),
       (:Debug, "----"),
       (:InBounds, "-"),
       (:Vectorize, ""))
for T in (Float32, Float64)
    dims = (10_000,)
    x = rand(T, dims)
    y = rand(T, dims)
    println()
    println("Tests for T=$T and $(length(x)) elements (\"Basic\" is ",
            "Julia own implementation):")
    for (P, str) in ops
        print(" - Time for `sum($P,x)` ", str, "----> ")
        @btime sum($P, $x)
    end
    println()
    for (P, str) in ops
        print(" - Time for `dot($P,x,y)` ", str, "----> ")
        @btime dot($P, $x, $y)
    end
end

end # module
