# Visible changes in `MayOptimize` package

## Version 0.3.0

* Methods `ldiv!`, `cholesky`, and `cholesky!` provided by the `LinearAlgebra`
  standard package have been extended to take an optimization level and/or
  algorithm supplemental argument to execute regular Julia code (not BLAS) on a
  single thread and with the various optimization levels of `MayOptimize`.  For
  small matrices (a few hundreds for `ldiv!`, up to 200×200 for the Cholesky
  decomposition), the single thread optimized regular code is much faster than
  multi-threaded BLAS.

* **Changes in type hierarchy.** Unexported abstract type
  `MayOptimize.CodeChoice` is the abstract parent type for the different types
  implemented by the package `MayOptimize`. Derived types are:

  - Unexported `MayOptimize.Standard` singleton type is to specify that a
    standard Julia method must be used.  This type is purposely not supported
    by `MayOptimize` macros as they should not be used with it.  This type is
    not exported by default.

  - `OptimLevel` is an abstract type whose sub-types are used to compile
    Julia methods with specific optimizations thanks to the macros of the
    `MayOptimize` package.

## Version 0.2.1

* Code cleanup.


## Version 0.2.0

* First official release.
