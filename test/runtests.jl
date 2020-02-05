module ConditionallyOptimizeTests

using Test, LinearAlgebra
using ConditionallyOptimize

function vsum(::Type{P},
              x::AbstractVector{T}) where {T<:AbstractFloat,P<:OptimLevel}
    s = zero(T)
    @may_assume_inbounds P for i in eachindex(x)
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
        @test vsum(Debug, x) ≈ sum(x)
        @test vsum(InBounds, x) ≈ sum(x)
        @test vsum(Vectorize, x) ≈ sum(x)
        @test vdot(Debug, x, y) ≈ dot(x,y)
        @test vdot(InBounds, x, y) ≈ dot(x,y)
        @test vdot(Vectorize, x, y) ≈ dot(x,y)
    end
end

# These are is not real benchmarks.
for T in (Float32, Float64)
    x = rand(T, dims)
    y = rand(T, dims)
    println()
    println("Tests for T=$T and $(length(x)) elements:")
    println(" - Time for `sum(x)`: ")
    @time sum(x)
    for P in (:Debug, :InBounds, :Vectorize)
        println(" - Time for `vsum($P,x)`: ")
        @eval @time vsum($P, $x)
    end
    println()
    println(" - Time for `dot(x,y)`: ")
    @time dot(x, y)
    for P in (:Debug, :InBounds, :Vectorize)
        println(" - Time for `vdot($P,x,y)`: ")
        @eval @time vdot($P, $x, $y)
    end
end

end # module
nothing
