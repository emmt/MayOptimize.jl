module ConditionallyOptimizeTests

using Test, LinearAlgebra, BenchmarkTools
using ConditionallyOptimize

function sum_inbounds(::Type{P},
                      x::AbstractVector{T}) where {T<:AbstractFloat,P}
    s = zero(T)
    @may_assume_inbounds P for i in eachindex(x)
        s += x[i]
    end
    return s
end

function sum_vectorize(::Type{P},
                       x::AbstractVector{T}) where {T<:AbstractFloat,P}
    s = zero(T)
    @may_vectorize P for i in eachindex(x)
        s += x[i]
    end
    return s
end

function dot_vectorize(::Type{P},
                       x::AbstractVector{T},
                       y::AbstractVector{T}) where {T<:AbstractFloat,P}
    s = zero(T)
    @may_vectorize P for i in eachindex(x, y)
        s += x[i]*y[i]
    end
    return s
end

# Sub-module to check detection of errors.
module Bogus

abstract type Debug end
abstract type InBounds end
abstract type Vectorize end
import ConditionallyOptimize: OptimLevel, @may_assume_inbounds, @may_vectorize
import ..ConditionallyOptimizeTests: sum_inbounds, sum_vectorize

# Correct versions.
sum0_inbounds(::Type{P}, x::AbstractVector) where {P<:OptimLevel} =
    sum_inbounds(P, x)
sum0_vectorize(::Type{P}, x::AbstractVector) where {P<:OptimLevel} =
    sum_vectorize(P, x)

# Bogus versions based on sum_inbounds.
sum1_debug(x::AbstractVector)     = sum_inbounds(Debug, x)
sum1_inbounds(x::AbstractVector)  = sum_inbounds(InBounds, x)
sum1_vectorize(x::AbstractVector) = sum_inbounds(Vectorize, x)

# Bogus versions based on sum_vectorize.
sum2_debug(x::AbstractVector)     = sum_vectorize(Debug, x)
sum2_inbounds(x::AbstractVector)  = sum_vectorize(InBounds, x)
sum2_vectorize(x::AbstractVector) = sum_vectorize(Vectorize, x)

end # Bogus module


dims = (10_000,)

@testset "Macros" begin
    for T in (Float32, Float64)
        x = rand(T, dims)
        y = rand(T, dims)
        @test sum_inbounds(Debug, x) ≈ sum(x)
        @test sum_inbounds(InBounds, x) ≈ sum(x)
        @test sum_inbounds(Vectorize, x) ≈ sum(x)
        @test sum_vectorize(Debug, x) ≈ sum(x)
        @test sum_vectorize(InBounds, x) ≈ sum(x)
        @test sum_vectorize(Vectorize, x) ≈ sum(x)
        @test dot_vectorize(Debug, x, y) ≈ dot(x,y)
        @test dot_vectorize(InBounds, x, y) ≈ dot(x,y)
        @test dot_vectorize(Vectorize, x, y) ≈ dot(x,y)
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
        @test_throws ErrorException Bogus.sum1_debug(x) ≈ sum(x)
        @test_throws ErrorException Bogus.sum1_inbounds(x) ≈ sum(x)
        @test_throws ErrorException Bogus.sum1_vectorize(x) ≈ sum(x)
        @test_throws ErrorException Bogus.sum2_debug(x) ≈ sum(x)
        @test_throws ErrorException Bogus.sum2_inbounds(x) ≈ sum(x)
        @test_throws ErrorException Bogus.sum2_vectorize(x) ≈ sum(x)
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
        print(" - Time for `sum($P,x)` ", str, "----> ")
        @btime sum_vectorize($P, $x)
    end
    println()
    print(" - Time for `dot(x,y)` --------------> ")
    @btime dot($x, $y)
    for (P, str) in ops
        print(" - Time for `dot($P,x,y)` ", str, "----> ")
        @btime dot_vectorize($P, $x, $y)
    end
end

end # module
nothing
