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

`OptimLevel.AVX` is an abstract optimization level.  It is intended to use the
`@avx` macro from `LoopVectorization` if available, or `@simd` otherwise (this
is the reason to inherit from [`Vectorize`](@ref)).

For now, this type is not exported by default and using it with the
`MayOptimize` macros behaves as if [`Vectorize`](@ref) optimization level has
been selected.  Specialized methods have to be written to exploit the power of
`@avx` which can deal with several loop levels (unlike `@simd` which is limite
to the innermost level).

"""
abstract type AVX <: Vectorize end

"""

`MayOptimize.Basic` is a singleton type derived from `MayOptimize.OptimLevel`
and intended to use version of methods provided by Julia.  This type is
purposely not supported by `MayOptimize` macros as they should not be used with
it.

This type is not exported by default.

"""
struct Basic <: OptimLevel end
