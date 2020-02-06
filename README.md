# Conditionally optimize Julia code

| **License**                     | **Build Status**                                                | **Code Coverage**                                                   |
|:--------------------------------|:----------------------------------------------------------------|:--------------------------------------------------------------------|
| [![][license-img]][license-url] | [![][travis-img]][travis-url] [![][appveyor-img]][appveyor-url] | [![][coveralls-img]][coveralls-url] [![][codecov-img]][codecov-url] |

When writing high performance [Julia][julia-url] code, you may want to keep
a reference code that perform bound-checking, another version that assumes
valid indices (and thus avoid bound-checking) and perhaps a more heavily
optimized version that requires vectorization of some loops.  The
[ConditionallyOptimize][repository-url] package let you have the 3 variants
available with a *single* version of the code.


## Documentation

The usage of [ConditionallyOptimize][repository-url] is summarized in the
following short example:

```julia
using ConditionallyOptimize

function foo!(::Type{P}, x::AbstractArray{T}) where {T<:Real, P<:OptimLevel}
    s = zero(T)
    # Loop 1: compute the sum of values.
    @may_assume_inbounds P for i in eachindex(x)
        s += x[i]
    end
    # Loop 2: fill with sum of values.
    @may_vectorize P for i in eachindex(x)
        x[i] += s
    end
    return x, s
end
```

Note that the two above loops are preceded by the macros
`@may_assume_inbounds` and `@may_vectorize` which both take 2 arguments: a
parameter `P` and an expression or a block of code (the 2nd argument must
be a simple `for` loop for the `@may_vectorize` macro).

How is compiled the expression or the block of code is determined by the
type parameter `P`:

- `P <: Debug` for debugging or reference code that performs bound-checking
  and no vectorization.

- `P <: InBounds` for code that assumes valid indices and thus avoids
  bound-checking.

- `P <: Vectorize` for code that assumes valid indices and requires
  vectorization.

A block of code provided to the `@may_assume_inbounds` macro will be
compiled with bound-checking (and thus no vectorization) if `P <: Debug`
and without bound-checking (as if `@inbounds` was specified) if
`P <: InBounds`.  Since `Vectorize <: InBounds`, specifying `Vectorize`
in `@may_assume_inbounds` also avoid bound-checking.

A block of code provided to the `@may_vectorize` macro will be compiled
with bound-checking and no vectorization if `P <: Debug`, with no
bound-checking if `P <: InBounds` (as if `@inbounds` was specified) and
with no bound-checking and vectorization if `P <: Vectorize` (as if both
`@inbounds` and `@simd` were specified).

Hence which version of `foo!` is called is decided by Julia method
dispatcher according to the abstract types `Debug`, `InBounds` or
`Vectorize` exported by `ConditionallyOptimize`.  Calling:

```
foo!(Debug, x)
```

executes a version that checks bounds and does no vectorization, while
calling

```
foo!(InBounds, x)
```

executes a version that avoids bound checking (in the 2 loops) and finally
calling:

```
foo!(Vectorize, x)
```
executes a version that avoids bound checking (in the 2 loops) and vectorizes
the second loop.

It is easy to provide a default version so that other users need not have
to bother choosing which version to use.  For instance, assuming that you
have checked that your code has no issues with indexing but that
vectorization makes almost no difference, you may write:

```julia
foo!(x::AbstractArray{T}) where {T<:Real} = foo!(InBounds, x)
```

and decide later to change the default optimization level.


## Installation

In Julia, hit the `]` key to switch to the package manager REPL (you should
get a `... pkg>` prompt) and type:

```julia
add https://github.com/emmt/ConditionallyOptimize.jl
```

No other packages are needed.

[repository-url]:  https://github.com/emmt/ConditionallyOptimize.jl

[doc-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[doc-dev-url]: https://emmt.github.io/ConditionallyOptimize.jl/dev

[license-url]: ./LICENSE.md
[license-img]: http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat

[travis-img]: https://travis-ci.org/emmt/ConditionallyOptimize.jl.svg?branch=master
[travis-url]: https://travis-ci.org/emmt/ConditionallyOptimize.jl

[appveyor-img]: https://ci.appveyor.com/api/projects/status/github/emmt/ConditionallyOptimize.jl?branch=master
[appveyor-url]: https://ci.appveyor.com/project/emmt/ConditionallyOptimize-jl/branch/master

[coveralls-img]: https://coveralls.io/repos/emmt/ConditionallyOptimize.jl/badge.svg?branch=master&service=github
[coveralls-url]: https://coveralls.io/github/emmt/ConditionallyOptimize.jl?branch=master

[codecov-img]: http://codecov.io/github/emmt/ConditionallyOptimize.jl/coverage.svg?branch=master
[codecov-url]: http://codecov.io/github/emmt/ConditionallyOptimize.jl?branch=master

[julia-url]: https://julialang.org/
[julia-pkgs-url]: https://pkg.julialang.org/
