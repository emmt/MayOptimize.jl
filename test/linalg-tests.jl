module TestingMayOptimizeLinearAlgebraMethods

using LinearAlgebra
using MayOptimize
using Test

using MayOptimize:
    AVX

using MayOptimize.LinearAlgebraMethods:
    CholeskyLowerColumnwise,
    CholeskyLowerRowwiseI,
    CholeskyLowerRowwiseII,
    CholeskyUpperColumnwiseI,
    CholeskyUpperColumnwiseII,
    CholeskyUpperRowwise,
    exec!

# Create an array initiallized with NaN's.
nans(A::AbstractArray) = nans(eltype(A), size(A))
nans(::Type{T}, dims::Integer...) where {T} = nans(T, dims)
nans(::Type{T}, dims::Tuple{Vararg{Integer}}) where {T<:AbstractFloat} =
    fill!(Array{T}(undef, dims), T(NaN))
nans(::Type{Complex{T}}, dims::Tuple{Vararg{Integer}}) where {T<:AbstractFloat} =
    fill!(Array{Complex{T}}(undef, dims), Complex{T}(T(NaN),T(NaN)))

function check(pred::Function, A::AbstractMatrix)
    I, J = axes(A)
    for j ∈ J
        for i ∈ I
            pred(A[i,j], i, j) || return false
        end
    end
    return true
end

function check(pred::Function, A::AbstractMatrix, B::AbstractMatrix)
    @assert axes(A) == axes(B)
    I, J = axes(A)
    for j ∈ J
        for i ∈ I
            pred(A[i,j], B[i,j], i, j) || return false
        end
    end
    return true
end

@testset "Cholesky factorization" begin
    for T in (Float64, Complex{Float64})
        # Build a random Hermitian positive definite matrix.
        m, n = 1000, 30
        B = 2*rand(T, m, n) .- 1
        A = B'*B
        A = (A + A')/2
        C = cholesky(A)
        for opt in (Debug, InBounds, Vectorize, AVX)
            for Alg in (CholeskyLowerColumnwise,
                        CholeskyLowerRowwiseI,
                        CholeskyLowerRowwiseII)
                alg = Alg(opt)
                # Test out-of-place Cholesky factorization.  Checking the validity
                # of the decomposition and that the strict upper part of L has been
                # untouched (it is filled with NaN's).
                L = LowerTriangular(exec!(alg, nans(A), A))
                @test L*L' ≈ A
                @test L ≈ C.L
                @test check((a,i,j) -> (i ≥ j || isnan(a)), parent(L))
                # Idem for the in-place operation version.
                X = copy(A)
                L = LowerTriangular(exec!(alg, X))
                @test L*L' ≈ A
                @test L ≈ C.L
                @test check((a,b,i,j) -> (i ≥ j || a === b), A, X)
            end
            for Alg in (CholeskyUpperColumnwiseI,
                        CholeskyUpperColumnwiseII,
                        CholeskyUpperRowwise)
                alg = Alg(opt)
                # Test out-of-place Cholesky factorization.  Checking the validity
                # of the decomposition and that the strict lower part of R has been
                # untouched (it is filled with NaN's).
                R = UpperTriangular(exec!(alg, nans(A), A))
                @test R'*R ≈ A
                @test R ≈ C.U
                @test check((a,i,j) -> (i ≤ j || isnan(a)), parent(R))
                # Idem for the in-place operation version.
                X = copy(A)
                R = UpperTriangular(exec!(alg, X))
                @test R'*R ≈ A
                @test R ≈ C.U
                @test check((a,b,i,j) -> (i ≤ j || a === b), A, X)
            end
        end
    end
end

end # module
