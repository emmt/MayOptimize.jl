#
# linalg.jl -
#
# A few linear algebra algorithms optimized conditionally.
#
module LinearAlgebraMethods

using ..MayOptimize
using ..MayOptimize: AVX, Standard

using LinearAlgebra
using LinearAlgebra: AbstractTriangular
import LinearAlgebra: dot, ldiv!, lmul!, cholesky, cholesky!

using Base: has_offset_axes
import Base: sum

abstract type AbstractAlgorithm end
abstract type CholeskyFactorization{opt<:OptimLevel} <: AbstractAlgorithm end
abstract type CholeskyLower{opt} <: CholeskyFactorization{opt} end
abstract type CholeskyUpper{opt} <: CholeskyFactorization{opt} end

# There are 2 variants to perform the Cholesky factorization
# Cholesky-Banachiewicz (which proceeds row-size) and Cholesky-Banachiewicz
# (which proceeds column-size) and they can be used to compute the L⋅L' or the
# R'⋅R factorizations.
struct CholeskyBanachiewiczLowerII{opt} <: CholeskyLower{opt} end
struct CholeskyBanachiewiczLowerI{ opt} <: CholeskyLower{opt} end
struct CholeskyBanachiewiczUpper{  opt} <: CholeskyUpper{opt} end
struct CholeskyCroutLower{         opt} <: CholeskyLower{opt} end
struct CholeskyCroutUpperII{       opt} <: CholeskyUpper{opt} end
struct CholeskyCroutUpperI{        opt} <: CholeskyUpper{opt} end

"""
     AnyLowerTriangular{T}

is the union of lower triangular matrix types defined in Julia standard library
`LinearAlgebra` and wrapped over regular matrices, that is
`LowerTriangular{T,S}` and `UnitLowerTriangular{T,S}` with `S <:
AbstractMatrix{T}`.  The matrix (of type `S`) backing the storage of the
entries is given by calling the `parent` method.

For the parent, say `A`, of a lower-triangular matrix, only entries `A[i,j]`
for `i ≥ j` shall be accessed.

For the parent, say `A`, of a unit lower-triangular matrix, only entries
`A[i,j]` for `i > j` shall be accessed and `A[i,i] = 1` shall be assumed.

"""
const AnyLowerTriangular{T} = Union{
    LowerTriangular{T,<:AbstractMatrix{T}},
    UnitLowerTriangular{T,<:AbstractMatrix{T}}}

"""
     AnyUpperTriangular{T}

is the union of upper triangular matrix types defined in standard Julia library
`LinearAlgebra` and wrapped over regular matrices, that is
`UpperTriangular{T,S}` and `UnitUpperTriangular{T,S}` with `S <:
AbstractMatrix{T}`.  The matrix (of type `S`) backing the storage of the
entries is given by calling the `parent` method.

For the parent, say `A`, of an upper-triangular matrix, only entries `A[i,j]`
for `i ≤ j` shall be accessed.

For the parent, say `A`, of a unit upper-triangular matrix, only entries
`A[i,j]` for `i < j` shall be accessed and `A[i,i] = 1` shall be assumed.

"""
const AnyUpperTriangular{T} = Union{
    UpperTriangular{T,<:AbstractMatrix{T}},
    UnitUpperTriangular{T,<:AbstractMatrix{T}}}

default_optimization(::Type{<:AbstractAlgorithm}) = InBounds

for alg in (:CholeskyBanachiewiczLowerI,
            :CholeskyBanachiewiczLowerII,
            :CholeskyBanachiewiczUpper,
            :CholeskyCroutLower,
            :CholeskyCroutUpperI,
            :CholeskyCroutUpperII)
    @eval $alg(opt::Type{<:OptimLevel} = default_optimization($alg)) =
        $alg{opt}()
end

# Vectorization does not improve lower triangular Cholesky factorization, so
# make `InBounds` the default.
default_optimization(::Type{<:CholeskyLower}) = InBounds

# Vectorization does improve upper triangular Cholesky factorization, so make
# `Vectorize` the default.
default_optimization(::Type{<:CholeskyUpper}) = Vectorize

# Use the most efficient version by default.
const BestCholeskyFactorization = CholeskyBanachiewiczUpper
CholeskyFactorization() = BestCholeskyFactorization()
CholeskyFactorization(opt::Type{<:OptimLevel}) =
    BestCholeskyFactorization(opt)
CholeskyFactorization{opt}() where {opt<:OptimLevel} =
    BestCholeskyFactorization{opt}()

const Floats = Union{AbstractFloat,Complex{<:AbstractFloat}}

#------------------------------------------------------------------------------
# Some basic linear algebra methods.

sum(::Type{<:Standard}, x::AbstractArray) = sum(x)

function sum(opt::Type{<:OptimLevel}, x::AbstractArray)
    s = zero(eltype(x))
    @maybe_vectorized opt for i in eachindex(x)
        s += x[i]
    end
    return s
end

dot(::Type{<:Standard}, x::AbstractVector, y::AbstractVector) = dot(x, y)

function dot(opt::Type{<:OptimLevel},
             x::AbstractArray{Tx,N},
             y::AbstractArray{Ty,N}) where {Tx,Ty,N}
    s = zero(promote_type(Tx, Ty))
    @maybe_vectorized opt for i in eachindex(x, y)
        s += conj(x[i])*y[i]
    end
    return s
end

#------------------------------------------------------------------------------
# Left division by a matrix.
#
# Call standard methods.
#
ldiv!(::Type{<:Standard}, A, B) = ldiv!(A, B)
ldiv!(::Type{<:Standard}, Y, A, B) = ldiv!(Y, A, B)
if VERSION < v"1.4.0-rc1"
    # Generic fallback. This assumes that B and Y have the same sizes.
    ldiv!(Y::AbstractArray, A::AbstractMatrix, B::AbstractArray) = begin
        Y === B || copyto!(Y, B)
        return ldiv!(A, Y)
    end
end
#
# Store A\b in y for A triangular.
#
function ldiv!(opt::Type{<:OptimLevel},
               y::AbstractVector{T},
               A::AbstractTriangular{T},
               b::AbstractVector{T}) where {T<:Floats}
    y === b || copyto!(y, b)
    return ldiv!(opt, A, y)
end

#
# Store A\b in b for A upper triangular.
#
function ldiv!(opt::Type{<:OptimLevel},
               A::AnyUpperTriangular{T},
               b::AbstractVector{T}) where {T<:Floats}
    R = parent(A) # to avoid getindex overheads
    n = check_ldiv_args(R, b)
    @maybe_inbounds opt for j ∈ n:-1:1
        if b[j] != 0
            if !is_unit_triangular(A)
                b[j] /= R[j,j]
            end
            temp = b[j]
            @maybe_vectorized opt for i ∈ 1:j-1 # was j-1:-1:1 in BLAS
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
               A::AnyLowerTriangular{T},
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
                                            S<:AnyUpperTriangular{T}}
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

function ldiv!(opt::Type{<:OptimLevel},
               y::AbstractVector{T},
               A::Union{Adjoint{T,S},Transpose{T,S}},
               b::AbstractVector{T}) where {T<:Floats,
                                            S<:AnyUpperTriangular{T}}
    R = parent(parent(A)) # to avoid getindex overheads
    n = check_ldiv_args(y, R, b)
    f = conjugate_or_identity(A)
    @maybe_inbounds opt for j ∈ 1:n
        temp = b[j]
        @maybe_vectorized opt for i ∈ 1:j-1
            temp -= f(R[i,j])*y[i]
        end
        if !is_unit_triangular(A)
            temp /= f(R[j,j])
        end
        y[j] = temp
    end
    return y
end
#
# Store A'\b in b for A lower triangular.
#
function ldiv!(opt::Type{<:OptimLevel},
               A::Union{Adjoint{T,S},Transpose{T,S}},
               b::AbstractVector{T}) where {T<:Floats,
                                            S<:AnyLowerTriangular{T}}
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

function ldiv!(opt::Type{<:OptimLevel},
               y::AbstractVector{T},
               A::Union{Adjoint{T,S},Transpose{T,S}},
               b::AbstractVector{T}) where {T<:Floats,
                                            S<:AnyLowerTriangular{T}}
    L = parent(parent(A)) # to avoid getindex overheads
    n = check_ldiv_args(y, L, b)
    f = conjugate_or_identity(A)
    @maybe_inbounds opt for j ∈ n:-1:1
        temp = b[j]
        @maybe_vectorized opt for i ∈ n:-1:j+1
            temp -= f(L[i,j])*y[i]
        end
        if !is_unit_triangular(A)
            temp /= f(L[j,j])
        end
        y[j] = temp
    end
    return y
end

function ldiv!(opt::Type{<:OptimLevel},
               A::Cholesky{T},
               b::AbstractVector{T}) where{T}
    uplo = getfield(A, :uplo)
    if uplo === 'U'
        R = getfield(A, :factors)
        ldiv!(opt, R, ldiv!(opt, R', b))
    elseif uplo === 'L'
        L = getfield(A, :factors)
        ldiv!(opt, L', ldiv!(opt, L, b))
    else
        throw_bad_uplo_field(A)
    end
    return b
end

function ldiv!(opt::Type{<:OptimLevel},
               y::AbstractVector{T},
               A::Cholesky{T},
               b::AbstractVector{T}) where{T}
    uplo = getfield(A, :uplo)
    if uplo === 'U'
        R = getfield(A, :factors)
        ldiv!(opt, R, ldiv!(opt, y, R', b))
    elseif uplo === 'L'
        L = getfield(A, :factors)
        ldiv!(opt, L', ldiv!(opt, y, L, b))
    else
        throw_bad_uplo_field(A)
    end
    return b
end

@noinline throw_bad_uplo_field(A::Cholesky) = error(
    "Unexpected field uplo='", getfield(A, :uplo),
    "' in Cholesky factorization")

#------------------------------------------------------------------------------

"""
    exec!(alg, dst, args...) -> dst

executes algorithm `alg` with arguments `args...` saving the result in
destination `dst`.

""" exec!

# Cholesky factorization can be done in-place.
exec!(alg::CholeskyFactorization, A::AbstractMatrix) = exec!(alg, A, A)

# Call standard Cholesky method.
cholesky(::Type{<:Standard}, A::AbstractMatrix, args...; kwds...) =
    cholesky(A, args...; kwds...)
cholesky!(::Type{<:Standard}, A::AbstractMatrix, args...; kwds...) =
    cholesky!(A, args...; kwds...)

# Call default Cholesky method.
cholesky(opt::Type{<:OptimLevel}, A::AbstractMatrix) =
    cholesky(CholeskyFactorization(opt), A)
cholesky!(opt::Type{<:OptimLevel}, A::AbstractMatrix) =
    cholesky!(CholeskyFactorization(opt), A)
cholesky!(opt::Type{<:OptimLevel}, buf::AbstractMatrix, A::AbstractMatrix) =
    cholesky!(CholeskyFactorization(opt), buf, A)

# Call specific Cholesky method.
cholesky(alg::CholeskyFactorization, A::AbstractMatrix) =
    Cholesky(exec!(alg, similar(A), A), uplo_char(alg), 0)
cholesky!(alg::CholeskyFactorization, A::AbstractMatrix) =
    Cholesky(exec!(alg, A), uplo_char(alg), 0)
cholesky!(alg::CholeskyFactorization, buf::AbstractMatrix, A::AbstractMatrix) =
    Cholesky(exec!(alg, buf, A), uplo_char(alg), 0)

uplo_char(::CholeskyLower) = 'L'
uplo_char(::CholeskyUpper) = 'U'

#------------------------------------------------------------------------------
# Lower triangular Cholesky factorization.

# Cholesky–Banachiewicz algorithm (row-wise).
function exec!(alg::CholeskyBanachiewiczLowerI{opt},
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

# Improved version of Cholesky–Banachiewicz algorithm (row-wise).
function exec!(alg::CholeskyBanachiewiczLowerII{opt},
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

# Cholesky–Crout algorithm (column-wise).
function exec!(alg::CholeskyCroutLower{opt},
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

# Cholesky–Crout algorithm (column-wise).
function exec!(alg::CholeskyCroutUpperI{opt},
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

# Cholesky–Crout algorithm (column-wise) improved version.
function exec!(alg::CholeskyCroutUpperII{opt},
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

# Cholesky–Banachiewicz algorithm (row-wise).
function exec!(alg::CholeskyBanachiewiczUpper{opt},
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
    require_one_based_indexing(L, A)
    inds = axes(A)
    inds[1] == inds[2] || throw_dimension_mismatch(
        "expecting a square matrix")
    axes(L) == inds || throw_dimension_mismatch(
            "matrixes must have the same axes")
    return length(inds[1])
end

function check_ldiv_args(A::AbstractMatrix, b::AbstractVector)
    require_one_based_indexing(A, b)
    m, n = size(A)
    m == n || throw_dimension_mismatch(
        "A must be a square matrix")
    length(b) == n || throw_dimension_mismatch(
        "sizes of A and b are incompatible")
    return Int(n)
end

function check_ldiv_args(y::AbstractVector,
                         A::AbstractMatrix,
                         b::AbstractVector)
    require_one_based_indexing(y, A, b)
    m, n = size(A)
    m == n || throw_dimension_mismatch(
        "A must be a square matrix")
    length(y) == n || throw_dimension_mismatch(
        "sizes of A and y are incompatible")
    length(b) == n || throw_dimension_mismatch(
        "sizes of A and b are incompatible")
    return Int(n)
end

require_one_based_indexing(A...) =
    has_offset_axes(A...) && throw_argument_error(
        "arrays must have 1-based indexing")

throw_argument_error(msg::AbstractString) = throw(ArgumentError(msg))
@noinline throw_argument_error(args...) =
    throw_argument_error(string(args...))

throw_dimension_mismatch(msg::AbstractString) = throw(DimensionMismatch(msg))
@noinline throw_dimension_mismatch(args...) =
    throw_dimension_mismatch(string(args...))

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
