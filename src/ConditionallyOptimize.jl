module ConditionallyOptimize

export
    Debug,
    InBounds,
    OptimLevel,
    Vectorize,
    @may_assume_inbounds,
    @may_vectorize

"""

`OptimLevel` is the abstract parent type for the different optimization
levels implemented by the package *ConditionallyOptimize*.  Derived
types are:

- `Debug` for debugging or reference code that performs bound-checking and
  no vectorization.

- `InBounds` for code that assumes valid indices and thus avoids
  bound-checking.

- `Vectorize` for code that assumes valid indices and requires
  vectorization.

See macros [`@may_assume_inbounds`] and [`@may_vectorize`] for examples.

"""
abstract type OptimLevel end
abstract type Debug     <: OptimLevel end # reference code for debugging
abstract type InBounds  <: OptimLevel end # assume in-bounds
abstract type Vectorize <: InBounds   end # require vectorization (also assume in-bounds)

@doc @doc(OptimLevel) Debug
@doc @doc(OptimLevel) InBounds
@doc @doc(OptimLevel) Vectorize

"""

```julia
@may_assume_inbounds P blk
```

yields code (avoiding escaping for clarity):

```julia
if \$P <: InBounds
     @inbounds \$blk
else
     \$blk
fi
```

In words, bound-checking is turned on/off at compilation time depending on the
type `P` without the needs to explicitly duplicate the code.

A typical usage is to write a method like:

```julia
function vsum(::Type{P}, x::AbstractArray{T}) where {T<:Real, P<:OptimLevel}
    s = zero(T)
    @may_assume_inbounds P for i in eachindex(x)
        s += x[i]
    end
    return s
end
```

then `vsum(Debug,x)` will execute a version of the code that performs
bound-checking; while `vsum(InBounds,x)` or `vsum(Vectorize,x)` will
execute a version of the code that avoids bound-checking.

See [`@may_vectorize`], [`OptimLevel`].

"""
macro may_assume_inbounds(opt, blk)
    code = quote
        if $opt <: InBounds
            @inbounds $blk
        else
            $blk
        end
    end
    esc(code)
end

"""

```julia
@may_vectorize P blk
```

yields code (avoiding escaping for clarity):

```julia
if \$P <:Vectorize
     @inbounds @simd \$blk
elseif \$P <: InBounds
     @inbounds \$blk
else
     \$blk
fi
```

In words, vectorization and/or bound-checking are turned on/off at
compilation time depending on the type `P` without the needs to explicitly
duplicate the code.

A typical usage:

```julia
function vsum(::Type{P}, x::AbstractArray{T}) where {T<:Real, P<:OptimLevel}
    s = zero(T)
    @may_vectorize P for i in eachindex(x)
        s += x[i]
    end
    return s
end
```

then `vsum(Debug,x)` will execute a version of the code that performs
bound-checking and no vectorization; `vsum(InBounds,x)` will execute a version
of the code that avoids bound-checking (and may vectorize code); finally
`vsum(Vectorize,x)` will execute a version of the code that avoids
bound-checking and requires vectorization.

See [`@may_assume_inbounds`], [`OptimLevel`].

"""
macro may_vectorize(opt, blk)
    code = quote
        if $opt <: Vectorize
            @inbounds @simd $blk
        elseif $opt <: InBounds
            @inbounds $blk
        else
            $blk
        end
    end
    esc(code)
end

end # module
