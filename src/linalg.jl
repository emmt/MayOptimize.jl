#
# linalg.jl -
#
# A few linear algebra algorithms optimized conditionally.
#
module LinearAlgebraMethods

using ..MayOptimize
using ..MayOptimize: AVX, Basic

using LinearAlgebra
import LinearAlgebra: dot, ldiv!, lmul!

import Base: sum

abstract type AbstractAlgorithm end
abstract type CholeskyFactorization{opt<:OptimLevel} <: AbstractAlgorithm end
struct CholeskyLowerColumnwise{  opt} <: CholeskyFactorization{opt} end
struct CholeskyLowerRowwiseI{    opt} <: CholeskyFactorization{opt} end
struct CholeskyLowerRowwiseII{   opt} <: CholeskyFactorization{opt} end
struct CholeskyUpperColumnwiseI{ opt} <: CholeskyFactorization{opt} end
struct CholeskyUpperColumnwiseII{opt} <: CholeskyFactorization{opt} end
struct CholeskyUpperRowwise{     opt} <: CholeskyFactorization{opt} end

# Vectorize does not improve lower triangular Cholesky factorization, so make
# InBounds the default.
for Alg in (:CholeskyLowerColumnwise,
            :CholeskyLowerRowwiseI,
            :CholeskyLowerRowwiseII)
    @eval $Alg(opt::Type{<:OptimLevel} = InBounds) = $Alg{opt}()
end

# Vectorize does improve upper triangular Cholesky factorization, so make it the
# default.
for Alg in (:CholeskyUpperColumnwiseI,
            :CholeskyUpperColumnwiseII,
            :CholeskyUpperRowwise)
    @eval $Alg(opt::Type{<:OptimLevel} = InBounds) = $Alg{opt}()
end

const CholeskyLower{opt} = Union{CholeskyLowerColumnwise{opt},
                                 CholeskyLowerRowwiseI{opt},
                                 CholeskyLowerRowwiseII{opt}}

const CholeskyUpper{opt} = Union{CholeskyUpperColumnwiseI{opt},
                                 CholeskyUpperColumnwiseII{opt},
                                 CholeskyUpperRowwise{opt}}

const Floats = Union{AbstractFloat,Complex{<:AbstractFloat}}

#------------------------------------------------------------------------------
# Some basic linear algebra methods.

sum(::Type{<:Basic}, x::AbstractArray) = sum(x)

function sum(opt::Type{<:OptimLevel}, x::AbstractArray)
    s = zero(eltype(x))
    @maybe_vectorized opt for i in eachindex(x)
        s += x[i]
    end
    return s
end

dot(::Type{<:Basic}, x::AbstractVector, y::AbstractVector) = dot(x, y)

function dot(opt::Type{<:OptimLevel},
             x::AbstractArray{Tx,N},
             y::AbstractArray{Ty,N}) where {Tx,Ty,N}
    s = zero(promote_type(Tx, Ty))
    @maybe_vectorized opt for i in eachindex(x, y)
        s += x[i]*y[i]
    end
    return s
end

#------------------------------------------------------------------------------
# Left division by a matrix.
#
# Call standard methods.
#
ldiv!(::Type{<:Basic}, A, B) = ldiv!(A, B)
ldiv!(::Type{<:Basic}, Y, A, B) = ldiv!(Y, A, B)
#
# Store A\b in b for A upper triangular.
#
function ldiv!(opt::Type{<:OptimLevel},
               A::Union{UpperTriangular{T,<:AbstractMatrix{T}},
                        UnitUpperTriangular{T,<:AbstractMatrix{T}}},
               b::AbstractVector{T}) where {T<:Floats}
    R = parent(A) # to avoid getindex overheads
    n = check_ldiv_args(R, b)
    @maybe_inbounds opt for j ∈ n:-1:1
        if b[j] != 0
            if !is_unit_triangular(A)
                b[j] /= R[j,j]
            end
            temp = b[j]
            @maybe_vectorized opt for i ∈ j-1:-1:1
                b[i] -= temp*R[i,j]
            end
        end
    end
    return b
end
#
# Store A\b in b for A lower triangular.
#
function ldiv!(opt::Type{<:OptimLevel},
               A::Union{LowerTriangular{T,<:AbstractMatrix{T}},
                        UnitLowerTriangular{T,<:AbstractMatrix{T}}},
               b::AbstractVector{T}) where {T<:Floats}
    L = parent(A) # to avoid getindex overheads
    n = check_ldiv_args(L, b)
    @maybe_inbounds opt for j ∈ 1:n
        if b[j] != 0
            if !is_unit_triangular(A)
                b[j] /= L[j,j]
            end
            temp = b[j]
            @maybe_vectorized opt for i ∈ j+1:n
                b[i] -= temp*L[i,j]
            end
        end
    end
    return b
end
#
# Store A'\b in b for A upper triangular.
#
function ldiv!(opt::Type{<:OptimLevel},
               A::Union{Adjoint{T,S},Transpose{T,S}},
               b::AbstractVector{T}) where {T<:Floats,
                                            S<:Union{<:UpperTriangular{T},
                                                     <:UnitUpperTriangular{T}}}
    R = parent(parent(A)) # to avoid getindex overheads
    n = check_ldiv_args(R, b)
    f = conjugate_or_identity(A)
    @maybe_inbounds opt for j ∈ 1:n
        temp = b[j]
        @maybe_vectorized opt for i ∈ 1:j-1
            temp -= f(R[i,j])*b[i]
        end
        if !is_unit_triangular(A)
            temp /= f(R[j,j])
        end
        b[j] = temp
    end
    return b
end
#
# Store A'\b in b for A lower triangular.
#
function ldiv!(opt::Type{<:OptimLevel},
               A::Union{Adjoint{T,S},Transpose{T,S}},
               b::AbstractVector{T}) where {T<:Floats,
                                            S<:Union{<:LowerTriangular{T},
                                                     <:UnitLowerTriangular{T}}}
    L = parent(parent(A)) # to avoid getindex overheads
    n = check_ldiv_args(L, b)
    f = conjugate_or_identity(A)
    @maybe_inbounds opt for j ∈ n:-1:1
        temp = b[j]
        @maybe_vectorized opt for i ∈ n:-1:j+1
            temp -= f(L[i,j])*b[i]
        end
        if !is_unit_triangular(A)
            temp /= f(L[j,j])
        end
        b[j] = temp
    end
    return b
end

#------------------------------------------------------------------------------

"""
    exec!(alg, dst, args...) -> dst

executes algorithm `alg` with arguments `args...` saving the result in
destination `dst`.

""" exec!

# Cholesky factorization can be done in-place.
exec!(alg::CholeskyFactorization, A::AbstractMatrix) = exec!(alg, A, A)

#------------------------------------------------------------------------------
# Lower triangular Cholesky factorization.

function exec!(alg::CholeskyLowerRowwiseI{opt},
               L::AbstractMatrix{T},
               A::AbstractMatrix{T}) where {T<:Floats,opt}
    n = check_chol_args(L, A)
    @maybe_inbounds opt for i ∈ 1:n
        @maybe_inbounds opt for j ∈ 1:i-1
            L[i,j] = (A[i,j] - _pdot(alg,L,i,j))/conj(L[j,j])
        end
        s = _pdot(alg,L,i)
        L[i,i] = sqrt(A[i,i] - s)
    end
    return L
end

function exec!(alg::CholeskyLowerRowwiseII{opt},
               L::AbstractMatrix{T},
               A::AbstractMatrix{T}) where {T<:Floats,opt}
    n = check_chol_args(L, A)
    @maybe_inbounds opt for i ∈ 1:n
        s = zero(real(T))
        @maybe_inbounds opt for j ∈ 1:i-1
            L[i,j] = (A[i,j] - _pdot(alg,L,i,j))/conj(L[j,j])
            s += abs2(L[i,j])
        end
        L[i,i] = sqrt(A[i,i] - s)
    end
    return L
end

function exec!(alg::CholeskyLowerColumnwise{opt},
               L::AbstractMatrix{T},
               A::AbstractMatrix{T}) where {T<:Floats,opt}
    n = check_chol_args(L, A)
    @maybe_inbounds opt for j ∈ 1:n
        s = _pdot(alg,L,j)
        L[j,j] = sqrt(A[j,j] - s)
        @maybe_inbounds opt for i ∈ j+1:n
            L[i,j] = (A[i,j] - _pdot(alg,L,i,j))/conj(L[j,j])
        end
    end
    return L
end

#------------------------------------------------------------------------------
# Upper triangular Cholesky factorization.

function exec!(alg::CholeskyUpperColumnwiseI{opt},
               R::AbstractMatrix{T},
               A::AbstractMatrix{T}) where {T<:Floats,opt}
    n = check_chol_args(R, A)
    @maybe_inbounds opt for j ∈ 1:n
        @maybe_inbounds opt for i ∈ 1:j-1
            R[i,j] = (A[i,j] - _pdot(alg,R,i,j))/conj(R[i,i])
        end
        s = _pdot(alg,R,j)
        R[j,j] = sqrt(A[j,j] - s)
    end
    return R
end

function exec!(alg::CholeskyUpperColumnwiseII{opt},
               R::AbstractMatrix{T},
               A::AbstractMatrix{T}) where {T<:Floats,opt}
    # Proceed column by column.
    n = check_chol_args(R, A)
    @maybe_inbounds opt for j ∈ 1:n
        s = zero(real(T))
        @maybe_inbounds opt for i ∈ 1:j-1
            R[i,j] = (A[i,j] - _pdot(alg,R,i,j))/conj(R[i,i])
            s += abs2(R[i,j])
        end
        R[j,j] = sqrt(A[j,j] - s)
    end
    return R
end

function exec!(alg::CholeskyUpperRowwise{opt},
               R::AbstractMatrix{T},
               A::AbstractMatrix{T}) where {T<:Floats,opt}
    n = check_chol_args(R, A)
    @maybe_inbounds opt for i ∈ 1:n
        s = _pdot(alg,R,i)
        R[i,i] = sqrt(A[i,i] - s)
        @maybe_inbounds opt for j ∈ i+1:n
            R[i,j] = (A[i,j] - _pdot(alg,R,i,j))/conj(R[i,i])
        end
    end
    return R
end

#------------------------------------------------------------------------------
# Partial row/column dot product for the Cholesky factorization.

"""
    _pdot(alg::CholeskyUpper, R, i, j) -> sum(conj(R[k,i])*R[k,j] for k ∈ 1:i-1)

yields the partial dot product of the `i`-th and `j`-th columns of matrix `R`
assuming it is a strict upper triangular matrix (as if entries are zero in the
diagonal and in the lower triangular part of `R`).

The condition `i < j` must hold, this is not checked.  If `i = j`, then call:

    _pdot(alg::CholeskyUpper, R, i) -> sum(abs2(R[k,i]) for k ∈ 1:i-1)

which is real even though `eltype(R)` may be complex.

---

    _pdot(alg::CholeskyLower, L, i, j) -> sum(L[i,k]*conj(L[j,k]) for k ∈ 1:j-1)

computes the partial dot product of the `i`-th and `j`-th rows of matrix `L`
assuming it is a strict lower triangular matrix (as if entries are zero in the
diagonal and in the upper triangular part of `L`).

The condition `i > j` must hold, this is not checked.  If `i = j`, then call:

    _pdot(alg::CholeskyLower, L, i) -> sum(abs2(L[i,k]) for k ∈ 1:i-1)

which is real even though `eltype(L)` may be complex.

---

The method `_pdot` is meant to be fast.  It is compiled according to the
optimization level embedded in `alg`.  The method shall be considered as
*unsafe* as, for maximum performances, the validity of the indices `i` and `j`
is not checked.

"""
@inline function _pdot(::CholeskyUpper{opt},
                       R::AbstractMatrix{T},
                       i::Int, j::Int) where {T,opt}
    s = zero(T)
    @maybe_vectorized opt for k ∈ 1:i-1
        s += conj(R[k,i])*R[k,j]
    end
    return s
end

@inline function _pdot(::CholeskyUpper{opt},
                       R::AbstractMatrix{T},
                       i::Int) where {T,opt}
    s = zero(real(T))
    @maybe_vectorized opt for k ∈ 1:i-1
        s += abs2(R[k,i])
    end
    return s
end

@inline function _pdot(::CholeskyLower{opt},
                       L::AbstractMatrix{T},
                       i::Int, j::Int) where {T,opt}
    s = zero(T)
    @maybe_vectorized opt for k in 1:j-1
        s += L[i,k]*conj(L[j,k])
    end
    return s
end

@inline function _pdot(::CholeskyLower{opt},
                       L::AbstractMatrix{T},
                       i::Int) where {T,opt}
    s = zero(real(T))
    @maybe_vectorized opt for k in 1:i-1
        s += abs2(L[i,k])
    end
    return s
end

#------------------------------------------------------------------------------

function check_chol_args(L::AbstractMatrix, A::AbstractMatrix)
    Base.require_one_based_indexing(L, A)
    inds = axes(A)
    inds[1] == inds[2] ||
        throw(DimensionMismatch("expecting a square matrix"))
    axes(L) == inds ||
        throw(DimensionMismatch("matrixes must have the same axes"))
    return length(inds[1])
end

function check_ldiv_args(A::AbstractMatrix, b::AbstractVector)
    Base.require_one_based_indexing(A, b)
    m, n = size(A)
    m == n ||
        throw(DimensionMismatch("A must be a square matrix"))
    length(b) == n ||
        throw(DimensionMismatch("sizes of A and b are incompatible"))
    return Int(n)
end

"""
    is_unit_triangular(A)

yields whether the matrix `A` is a unit triangular matrix or not.  The result
is type-stable and is directly inferred from the type of `A`.  The argument can
also be a matrix type.

"""
is_unit_triangular(::T) where {T<:AbstractMatrix} = is_unit_triangular(T)
is_unit_triangular(::Type{<:Adjoint{T,S}}) where {T,S} = is_unit_triangular(S)
is_unit_triangular(::Type{<:Transpose{T,S}}) where {T,S} = is_unit_triangular(S)
is_unit_triangular(::Type{<:UnitLowerTriangular}) = true
is_unit_triangular(::Type{<:UnitUpperTriangular}) = true
is_unit_triangular(::Type{<:AbstractMatrix}) = false

"""
    conjugate_or_identity(A)

yields `conj` if multiplying or dividing by the matrix `A` requires to take the
conjugate of the values of the object backing the storage of the entries of
`A`; otherwise yields `identity`.

"""
conjugate_or_identity(::Adjoint{<:Real}) = identity
conjugate_or_identity(::Adjoint{<:Complex}) = conj
conjugate_or_identity(::Transpose{<:Real}) = identity
conjugate_or_identity(::Transpose{<:Complex}) = identity
conjugate_or_identity(::AbstractMatrix) = identity

end # module
