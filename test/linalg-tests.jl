module TestingMayOptimizeLinearAlgebraMethods

using LinearAlgebra
using MayOptimize
using Test

using MayOptimize:
    Standard,
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

# Generate random values uniformly distributed in [-1,1].
generate_array(T::Type{<:Real}, dims...) = 2*rand(T, dims...) .- 1
generate_array(T::Type{<:Complex{<:Real}}, dims...) =
    2*rand(T, dims...) .- Complex(1,1)

# Build a random Hermitian positive definite matrix.
function generate_symmetric_definite_matrix(T::Type, n::Integer)
    H = generate_array(T, 10n, n)
    A = H'*H
    return (A + A')/2
end

@testset "ldiv!(opt, [y,] A, b) " begin
    for T in (Float32, Float64, Complex{Float64})
        n = 30
        A = generate_symmetric_definite_matrix(T, n)
        C = cholesky(A)
        L = C.L
        @test L*L' ≈ A
        R = C.U
        @test R'*R ≈ A
        w = Array{T}(undef, n)
        x = 2*rand(T, n) .- 1
        y = A*x
        z = Array{T}(undef, n)
        for opt in (Standard, Debug, InBounds, Vectorize)
            # ldiv! on lower triangular matrix L.
            @test ldiv!(opt, L, copyto!(z, y)) ≈ L\y
            @test ldiv!(opt, z, L, y) ≈ L\y
            @test ldiv!(opt, L', copyto!(w, z)) ≈ L'\z
            @test ldiv!(opt, w, L', z) ≈ L'\z
            @test ldiv!(opt, L', ldiv!(opt, L, copyto!(z, y))) ≈ x
            @test ldiv!(opt, w, L', ldiv!(opt, z, L, y)) ≈ x
            #@test lmul!(opt, L, lmul!(opt, L', copyto!(z, x))) ≈ y
            #@test lmul!(opt, L, lmul!(opt, z, L', x)) ≈ y
            # ldiv! on upper triangular matrix R.
            @test ldiv!(opt, R', copyto!(z, y)) ≈ R'\y
            @test ldiv!(opt, z, R', y) ≈ R'\y
            @test ldiv!(opt, R, copyto!(w, z)) ≈ R\z
            @test ldiv!(opt, w, R, z) ≈ R\z
            @test ldiv!(opt, R, ldiv!(opt, R', copyto!(z, y))) ≈ x
            @test ldiv!(opt, w, R, ldiv!(opt, z, R', y)) ≈ x
            #@test lmul!(opt, R', lmul!(opt, R, copyto!(z, x))) ≈ y
            #@test lmul!(opt, R', lmul!(opt, z, R, x)) ≈ y
        end
    end
end

@testset "Cholesky factorization" begin
    for T in (Float64, Complex{Float64})
        # Build a random Hermitian positive definite matrix.
        n = 30
        A = generate_symmetric_definite_matrix(T, n)
        C = cholesky(A)
        w = Array{T}(undef, n)
        x = 2*rand(T, n) .- 1
        y = A*x
        z = Array{T}(undef, n)
        Cs = cholesky(Standard, A)
        @test Cs.L == C.L
        @test Cs.U == C.U
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
