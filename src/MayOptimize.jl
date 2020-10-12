module MayOptimize

export
    Debug,
    InBounds,
    OptimLevel,
    Vectorize,
    @maybe_inbounds,
    @maybe_vectorized

"""

`OptimLevel` is the abstract parent type for the different optimization levels
implemented by the package *MayOptimize*.  Derived types are:

- `Debug` for debugging or reference code that performs bound checking and no
  vectorization.

- `InBounds` for code that assumes valid indices and thus avoids
  bound checking.

- `Vectorize` for code that assumes valid indices and requires vectorization.

See macros [`@maybe_inbounds`] and [`@maybe_vectorized`] for examples.

"""
abstract type OptimLevel end
abstract type Debug     <: OptimLevel end # reference code for debugging
abstract type InBounds  <: OptimLevel end # assume in-bounds
abstract type Vectorize <: InBounds   end # to vectorize (also assume in-bounds)
@doc @doc(OptimLevel) Debug
@doc @doc(OptimLevel) InBounds
@doc @doc(OptimLevel) Vectorize

"""

    @maybe_inbounds P blk

yields code that may compile block of code `blk` without bound checking if
allowed by `P`, that is if `P` inherits from `InBounds`.  This is equivalent to
(without escaping for clarity):

```julia
if \$P <: InBounds
     @inbounds \$blk
else
     \$blk
end
```

In words, bound checking is turned on/off at compilation time depending on the
type `P` without the needs to explicitly duplicate the code.

A typical usage is to write a method like:

```julia
function vsum(::Type{P}, x::AbstractArray{T}) where {T<:Real, P<:OptimLevel}
    s = zero(T)
    @maybe_inbounds P for i in eachindex(x)
        s += x[i]
    end
    return s
end
```

then `vsum(Debug,x)` will execute a version of the code that performs bound
checking; while `vsum(InBounds,x)` or `vsum(Vectorize,x)` will execute a
version of the code that avoids bound checking.

See [`@maybe_vectorized`], [`OptimLevel`].

"""
macro maybe_inbounds(P, blk)
    # See comments in `@maybe_vectorized`.
    opt = esc(:($P))
    blk0 = esc(:($blk))
    blk1 = esc(:(@inbounds $blk))
    quote
        if $opt <: InBounds
            $blk1
        elseif $opt <: Debug
            $blk0
        else
            error("argument `P` of macro `@maybe_inbounds` is not a valid optimization level")
        end
    end
end

"""
    @maybe_vectorized P blk

yields code that may compile block of code `blk` with vectorization and/or
without bound checking as allowed by the optimization level `P`.  This is
equivalent to (without escaping for clarity):

```julia
if \$P <:Vectorize
     @inbounds @simd \$blk
elseif \$P <: InBounds
     @inbounds \$blk
else
     \$blk
end
```

In words, vectorization and/or bound checking are turned on/off at compilation
time depending on the type `P` without the needs to explicitly duplicate the
code.

A typical usage:

```julia
function vsum(::Type{P}, x::AbstractArray{T}) where {T<:Real, P<:OptimLevel}
    s = zero(T)
    @maybe_vectorized P for i in eachindex(x)
        s += x[i]
    end
    return s
end
```

then `vsum(Debug,x)` will execute a version of the code that performs bound
checking and no vectorization; `vsum(InBounds,x)` will execute a version of the
code that avoids bound checking (and may vectorize code); finally
`vsum(Vectorize,x)` will execute a version of the code that avoids bound
checking and requires vectorization.

See [`@maybe_inbounds`], [`OptimLevel`].

"""
macro maybe_vectorized(P, blk)
    # Type to compare `P` with is to be interpreted in this module context
    # and thus must not be escaped.  But `opt` and `blk` must be escaped.
    # I empirically found that `esc(:(@macro1 @macro2 ... $expr))` was the
    # correct way to escape expression `expr`, possibly preceded by some
    # other macro calls; the alternative `:(esc($expr))` works for all
    # expressions below but not for the one with `@simd` because this macro
    # expects a `for` loop.
    opt = esc(:($P))
    blk0 = esc(:($blk))
    blk1 = esc(:(@inbounds $blk))
    blk2 = esc(:(@inbounds @simd $blk))
    quote
        if $opt <: Vectorize
            $blk2
        elseif $opt <: InBounds
            $blk1
        elseif $opt <: Debug
            $blk0
        else
            error("argument `P` of macro `@maybe_vectorized` is not a valid optimization level")
        end
    end
end

end # module
