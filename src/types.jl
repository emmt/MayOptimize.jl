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

# Optimization level to use @avx from LoopVectorization if available, @simd
# otherwise (this is the reason to inherit from Vectorize).
abstract type AVX <: Vectorize end
