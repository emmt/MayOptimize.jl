module ConditionallyOptimizeTests

using Test, LinearAlgebra, BenchmarkTools
using ConditionallyOptimize

# This version is to check @may_assume_inbounds.
function vsum1(::Type{P},
               x::AbstractVector{T}) where {T<:AbstractFloat,P<:OptimLevel}
    s = zero(T)
    @may_assume_inbounds P for i in eachindex(x)
        s += x[i]
    end
    return s
end

function vsum(::Type{P},
              x::AbstractVector{T}) where {T<:AbstractFloat,P<:OptimLevel}
    s = zero(T)
    @may_vectorize P for i in eachindex(x)
        s += x[i]
    end
    return s
end

function vdot(::Type{P},
              x::AbstractVector{T},
              y::AbstractVector{T}) where {T<:AbstractFloat,P<:OptimLevel}
    s = zero(T)
    @may_vectorize P for i in eachindex(x, y)
        s += x[i]*y[i]
    end
    return s
end

dims = (10_000,)

@testset "Macros" begin
    for T in (Float32, Float64)
        x = rand(T, dims)
        y = rand(T, dims)
        @test vsum1(Debug, x) ≈ sum(x)
        @test vsum1(InBounds, x) ≈ sum(x)
        @test vsum1(Vectorize, x) ≈ sum(x)
        @test vsum(Debug, x) ≈ sum(x)
        @test vsum(InBounds, x) ≈ sum(x)
        @test vsum(Vectorize, x) ≈ sum(x)
        @test vdot(Debug, x, y) ≈ dot(x,y)
        @test vdot(InBounds, x, y) ≈ dot(x,y)
        @test vdot(Vectorize, x, y) ≈ dot(x,y)
    end
end

ops = ((:Debug, "----"),
       (:InBounds, "-"),
       (:Vectorize, ""))
for T in (Float32, Float64)
    x = rand(T, dims)
    y = rand(T, dims)
    println()
    println("Tests for T=$T and $(length(x)) elements:")
    print(" - Time for `sum(x)` --------------> ")
    @btime sum($x)
    for (P, str) in ops
        print(" - Time for `vsum($P,x)` ", str, "---> ")
        @btime vsum($P, $x)
    end
    println()
    print(" - Time for `dot(x,y)` --------------> ")
    @btime dot($x, $y)
    for (P, str) in ops
        print(" - Time for `vdot($P,x,y)` ", str, "---> ")
        @btime vdot($P, $x, $y)
    end
end

end # module
nothing
