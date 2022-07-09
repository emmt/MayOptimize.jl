module TestingMayOptimizeLinearAlgebraMethods

export ldiv_tests, cholesky_tests

using Test
using LinearAlgebra
using LinearAlgebra: AbstractTriangular
using MayOptimize
using MayOptimize:
    CodeChoice,
    Standard,
    AVX
using MayOptimize.LinearAlgebraMethods:
    AbstractAlgorithm,
    CholeskyBanachiewiczLower,
    CholeskyBanachiewiczLowerI,
    CholeskyBanachiewiczLowerII,
    CholeskyBanachiewiczUpper,
    CholeskyCroutLower,
    CholeskyCroutUpper,
    CholeskyCroutUpperI,
    CholeskyCroutUpperII,
    CholeskyFactorization,
    CholeskyLower,
    CholeskyUpper,
    Floats,
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

function ldiv_tests(;
                    types = (Float32,
                             Float64,
                             Complex{Float64},),
                    sizes = (30,),
                    opts  = (Standard,
                             Debug,
                             InBounds,
                             Vectorize,))
    @testset "ldiv!(opt, [y,] A, b) " begin
        for T in types, n in sizes
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
            for opt in opts
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
end

function cholesky_tests(;
                        algs  = (CholeskyCroutLower,
                                 CholeskyBanachiewiczLowerI,
                                 CholeskyBanachiewiczLowerII,
                                 CholeskyCroutUpperI,
                                 CholeskyCroutUpperII,
                                 CholeskyBanachiewiczUpper,),
                        types = (Float64,
                                 Complex{Float64},),
                        sizes = (30,),
                        opts  = (Debug,
                                 InBounds,
                                 Vectorize,))
    @testset "Cholesky factorization" begin
        for T in types, n in sizes
            # Build a random Hermitian positive definite matrix.
            A = generate_symmetric_definite_matrix(T, n)
            C = cholesky(A)
            w = Array{T}(undef, n)
            x = 2*rand(T, n) .- 1
            y = A*x
            z = Array{T}(undef, n)
            std = false
            for opt in (Standard, opts...)
                if opt === Standard
                    if std
                        continue
                    else
                        std = true
                    end
                end
                Cs = cholesky(opt, A)
                if opt === Standard
                    @test Cs.L == C.L
                    @test Cs.U == C.U
                else
                    @test Cs.L ≈ C.L
                    @test Cs.U ≈ C.U
                end
                @test ldiv!(opt, z, Cs, y) ≈ x
                @test ldiv!(opt, Cs, copyto!(z, y)) ≈ x
            end
            for opt in opts, algtype in algs
                if algtype <: CholeskyLower
                    alg = algtype(opt)
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
                elseif algtype <: CholeskyUpper
                    alg = algtype(opt)
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
end

end # module
