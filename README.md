# Conditionally optimize Julia code

| **License**                     | **Build Status**                                                | **Code Coverage**                                                   |
|:--------------------------------|:----------------------------------------------------------------|:--------------------------------------------------------------------|
| [![][license-img]][license-url] | [![][travis-img]][travis-url] [![][appveyor-img]][appveyor-url] | [![][coveralls-img]][coveralls-url] [![][codecov-img]][codecov-url] |

When writing high performance [Julia][julia-url] code, you may want to keep a
reference code that perform bound checking, another version that assumes valid
indices (and thus avoid bound checking) and perhaps a more heavily optimized
version that requires loop vectorization.  The [MayOptimize][repository-url]
package let you have the 3 variants available with a *single* version of the
code.


## Documentation

The usage of [MayOptimize][repository-url] is summarized in the following short
example:

```julia
using MayOptimize

function foo!(::Type{P}, x::AbstractArray{T}) where {T<:Real, P<:OptimLevel}
    s = zero(T)
    # Loop 1: compute the sum of values.
    @maybe_inbounds P for i in eachindex(x)
        s += x[i]
    end
    # Loop 2: fill with sum of values.
    @maybe_vectorized P for i in eachindex(x)
        x[i] += s
    end
    return x, s
end
```

Note that the two above loops are preceded by the macros `@maybe_inbounds` and
`@maybe_vectorized` which both take 2 arguments: a parameter `P` and an
expression or a block of code (the 2nd argument must be a simple `for` loop for
the `@maybe_vectorized` macro).

How is compiled the expression or the block of code is determined by the
type parameter `P`:

- `P <: Debug` for debugging or reference code that performs bound checking and
  no vectorization.

- `P <: InBounds` for code that assumes valid indices and thus avoids bound
  checking.

- `P <: Vectorize` for code that assumes valid indices and requires
  vectorization.

A block of code provided to the `@maybe_inbounds` macro will be compiled with
bound checking (and thus no vectorization) if `P <: Debug` and without bound
checking (as if `@inbounds` was specified) if `P <: InBounds`.  Since
`Vectorize <: InBounds`, specifying `Vectorize` in `@maybe_inbounds` also avoid
bound checking.

A block of code provided to the `@maybe_vectorized` macro will be compiled with
bound checking and no vectorization if `P <: Debug`, with no bound checking if
`P <: InBounds` (as if `@inbounds` was specified) and with no bound checking
and vectorization if `P <: Vectorize` (as if both `@inbounds` and `@simd` were
specified).

Hence which version of `foo!` is called is decided by Julia method dispatcher
according to the abstract types `Debug`, `InBounds` or `Vectorize` exported by
`MayOptimize`.  Calling:

```julia
foo!(Debug, x)
```

executes a version that checks bounds and does no vectorization, while calling:

```julia
foo!(InBounds, x)
```

executes a version that avoids bound checking (in the 2 loops) and finally
calling:

```julia
foo!(Vectorize, x)
```
executes a version that avoids bound checking (in the 2 loops) and vectorizes
the second loop.

It is easy to provide a default version so that other users need not have to
bother choosing which version to use.  For instance, assuming that you have
checked that your code has no issues with indexing but that vectorization makes
almost no difference, you may write:

```julia
foo!(x::AbstractArray{T}) where {T<:Real} = foo!(InBounds, x)
```

and decide later to change the default optimization level.


## Installation

In Julia, hit the `]` key to switch to the package manager REPL (you should get
a `... pkg>` prompt) and type:

```julia
add MayOptimize
```

No other packages are needed.


## Examples

`MayOptimize` extends a few base linear algebra methods such as the `ldiv!`
method which performs left division by a matrix `A` and can be called as:

```julia
using MayOptimize, LinearAlgebra
ldiv!(opt, A, b)
ldiv!(opt, y, A, b)
```

In the first case, the operation is done in-place and `b` is overwritten with
`A\b`, in the second case, `A\b` is stored in `y`.  Argument `opt` can be
`MayOptimize.Standard` to use Julia standard method (probably BLAS), `Debug`,
`InBounds` or `Vectorize` to compile Julia code in `MayOptimize` with different
optimization settings.  The following figures (obtained with Julia 1.5.3 on an
AMD Ryzen Threadripper 2950X 16-Core processor) show how efficient can be Julia
code when compiled with well chosen optimization settings (note the 1.7 gain
compared to the standard implementation when `@simd` is used in the innermost
loop level).  Having a look at [`src/linalg.jl`](src/linalg.jl), you can
realize that the code is identical for the `Debug`, `InBounds` or `Vectorize`
settings (only the `opt` argument changes) and that this code turns out to be
pretty straightforward.

![Left division by a lower triangular matrix](figs/ldiv-L-median.png "")

![Left division by the transpose of a lower triangular matrix](figs/ldiv-Lt-median.png "")

![Left division by an upper triangular matrix](figs/ldiv-R-median.png "")

![Left division by the transpose of an upper triangular matrix](figs/ldiv-Rt-median.png "")


[repository-url]:  https://github.com/emmt/MayOptimize.jl

[doc-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[doc-dev-url]: https://emmt.github.io/MayOptimize.jl/dev

[license-url]: ./LICENSE.md
[license-img]: http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat

[travis-img]: https://travis-ci.org/emmt/MayOptimize.jl.svg?branch=master
[travis-url]: https://travis-ci.org/emmt/MayOptimize.jl

[appveyor-img]: https://ci.appveyor.com/api/projects/status/github/emmt/MayOptimize.jl?branch=master
[appveyor-url]: https://ci.appveyor.com/project/emmt/MayOptimize-jl/branch/master

[coveralls-img]: https://coveralls.io/repos/emmt/MayOptimize.jl/badge.svg?branch=master&service=github
[coveralls-url]: https://coveralls.io/github/emmt/MayOptimize.jl?branch=master

[codecov-img]: http://codecov.io/github/emmt/MayOptimize.jl/coverage.svg?branch=master
[codecov-url]: http://codecov.io/github/emmt/MayOptimize.jl?branch=master

[julia-url]: https://julialang.org/
[julia-pkgs-url]: https://pkg.julialang.org/
