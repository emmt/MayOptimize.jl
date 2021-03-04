module BenchmarkingMayOptimizeMacros

using BenchmarkTools, LinearAlgebra
using MayOptimize

struct Basic end
Base.sum(::Type{Basic}, x::AbstractArray) = sum(x)

function Base.sum(::Type{P},
                  x::AbstractArray{<:AbstractFloat}) where {P<:OptimLevel}
    s = zero(eltype(x))
    @maybe_vectorized P for i in eachindex(x)
        s += x[i]
    end
    return s
end

LinearAlgebra.dot(::Type{Basic}, x::AbstractVector, y::AbstractVector) = dot(x, y)

function LinearAlgebra.dot(::Type{P},
                           x::AbstractVector{T},
                           y::AbstractVector{T}) where {T<:AbstractFloat,
                                                        P<:OptimLevel}
    s = zero(T)
    @maybe_vectorized P for i in eachindex(x, y)
        s += x[i]*y[i]
    end
    return s
end

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
