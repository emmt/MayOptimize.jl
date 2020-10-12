module MayOptimizeTests

using Test, LinearAlgebra, BenchmarkTools
using MayOptimize

# Singleton type to use version provided by Julia.
struct Basic end
struct Dummy <: MayOptimize.OptimLevel end

function sum_inbounds(::Type{P},
                      x::AbstractVector{T}) where {T<:AbstractFloat,
                                                   P<:OptimLevel}
    s = zero(T)
    @maybe_inbounds P for i in eachindex(x)
        s += x[i]
    end
    return s
end

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

# Sub-module to check detection of errors.
module Bogus

abstract type Debug end
abstract type InBounds end
abstract type Vectorize end
import MayOptimize: OptimLevel, @maybe_inbounds, @maybe_vectorized
import ..MayOptimizeTests: sum_inbounds

# Correct versions.
sum0_inbounds(::Type{P}, x::AbstractVector) where {P<:OptimLevel} =
    sum_inbounds(P, x)
sum0_vectorize(::Type{P}, x::AbstractVector) where {P<:OptimLevel} =
    sum(P, x)

# Bogus versions based on sum_inbounds.
sum1_debug(x::AbstractVector)     = sum_inbounds(Debug, x)
sum1_inbounds(x::AbstractVector)  = sum_inbounds(InBounds, x)
sum1_vectorize(x::AbstractVector) = sum_inbounds(Vectorize, x)

# Bogus versions based on sum.
sum2_debug(x::AbstractVector)     = sum(Debug, x)
sum2_inbounds(x::AbstractVector)  = sum(InBounds, x)
sum2_vectorize(x::AbstractVector) = sum(Vectorize, x)

end # Bogus module


dims = (10_000,)

@testset "Macros" begin
    for T in (Float32, Float64)
        x = rand(T, dims)
        y = rand(T, dims)
        @test_throws ErrorException sum(Dummy, x) ≈ sum(x)
        @test_throws ErrorException sum_inbounds(Dummy, x) ≈ sum(x)
        @test sum_inbounds(Debug, x) ≈ sum(x)
        @test sum_inbounds(InBounds, x) ≈ sum(x)
        @test sum_inbounds(Vectorize, x) ≈ sum(x)
        @test sum(Basic, x) == sum(x)
        @test sum(Debug, x) ≈ sum(x)
        @test sum(InBounds, x) ≈ sum(x)
        @test sum(Vectorize, x) ≈ sum(x)
        @test dot(Basic, x, y) == dot(x,y)
        @test dot(Debug, x, y) ≈ dot(x,y)
        @test dot(InBounds, x, y) ≈ dot(x,y)
        @test dot(Vectorize, x, y) ≈ dot(x,y)
        # These ones should work.
        @test Bogus.sum0_inbounds(Debug, x) ≈ sum(x)
        @test Bogus.sum0_inbounds(InBounds, x) ≈ sum(x)
        @test Bogus.sum0_inbounds(Vectorize, x) ≈ sum(x)
        @test Bogus.sum0_vectorize(Debug, x) ≈ sum(x)
        @test Bogus.sum0_vectorize(InBounds, x) ≈ sum(x)
        @test Bogus.sum0_vectorize(Vectorize, x) ≈ sum(x)
        # These ones should fail, although for different reasons.
        @test_throws MethodError Bogus.sum0_inbounds(Bogus.Debug, x) ≈ sum(x)
        @test_throws MethodError Bogus.sum0_inbounds(Bogus.InBounds, x) ≈ sum(x)
        @test_throws MethodError Bogus.sum0_inbounds(Bogus.Vectorize, x) ≈ sum(x)
        @test_throws MethodError Bogus.sum0_vectorize(Bogus.Debug, x) ≈ sum(x)
        @test_throws MethodError Bogus.sum0_vectorize(Bogus.InBounds, x) ≈ sum(x)
        @test_throws MethodError Bogus.sum0_vectorize(Bogus.Vectorize, x) ≈ sum(x)
        @test_throws MethodError Bogus.sum1_debug(x) ≈ sum(x)
        @test_throws MethodError Bogus.sum1_inbounds(x) ≈ sum(x)
        @test_throws MethodError Bogus.sum1_vectorize(x) ≈ sum(x)
        @test_throws MethodError Bogus.sum2_debug(x) ≈ sum(x)
        @test_throws MethodError Bogus.sum2_inbounds(x) ≈ sum(x)
        @test_throws MethodError Bogus.sum2_vectorize(x) ≈ sum(x)
    end
end

ops = ((:Basic, "----"),
       (:Debug, "----"),
       (:InBounds, "-"),
       (:Vectorize, ""))
for T in (Float32, Float64)
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
nothing
