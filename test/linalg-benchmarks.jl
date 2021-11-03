module MayOptimizeLinearAlgebraBenchmarks

export
    run_benchmarks,
    load_benchmarks,
    save_benchmarks

const USE_AVX = false

using HDF5
using LinearAlgebra
using Statistics
using StaticArrays

using BenchmarkTools
using BenchmarkTools: Trial

using MayOptimize
using MayOptimize:
    AVX,
    Standard,
    CholeskyBanachiewiczLowerI,
    CholeskyBanachiewiczLowerII,
    CholeskyBanachiewiczUpper,
    CholeskyCroutLower,
    CholeskyCroutUpperI,
    CholeskyCroutUpperII,
    exec!

using MayOptimize.LinearAlgebraMethods:
    AbstractAlgorithm,
    CholeskyFactorization,
    Floats

import MayOptimize.LinearAlgebraMethods:
    exec!

struct LDiv{opt} <: AbstractAlgorithm
    operand::String
end

exec!(alg::LDiv{opt}, args...) where {opt} = ldiv!(opt, args...)

struct CholeskyBLAS <: AbstractAlgorithm end
struct CholeskyStatic <: AbstractAlgorithm end

group(T::Type{<:Floats}, alg::Type{<:AbstractAlgorithm}) = group(T, alg())

@noinline group(T::Type{<:Floats}, alg::AbstractAlgorithm) =
    string(T, "/", group(alg))

@noinline group(alg::LDiv{<:Standard}) = "ldiv!-$(alg.operand)/BLAS"
@noinline group(alg::LDiv{opt}) where {opt} =
    string("ldiv!-", alg.operand, "/", nameof(opt))

group(::CholeskyBLAS  ) = "Cholesky/BLAS"
group(::CholeskyStatic) = "Cholesky/StaticArrays"

const CholeskyMethod = Union{CholeskyFactorization,CholeskyBLAS,CholeskyStatic}
nops(::CholeskyMethod, n::Integer) = (n^3)/3

# Assume triangular matrix. Number of perations is: n divisions plus 1 - 2n +
# n² additions and multiplications, which is approximately n² (because
# divisions cost more than additions and multiplications).
nops(::LDiv, n::Integer) = n^2

for A in (:CholeskyBanachiewiczLowerII,
          :CholeskyBanachiewiczLowerI,
          :CholeskyBanachiewiczUpper,
          :CholeskyCroutLower,
          :CholeskyCroutUpperII,
          :CholeskyCroutUpperI)
    @eval @noinline group(::$A{opt}) where {opt} =
        string($(string(A)), "/", nameof(opt))
end

@static if USE_AVX
    include("../src/linalg-avx.jl")
end

# struct BenchmarkTools.Trial:
#  params  - BenchmarkTools.Parameters
#  times   - measured times in nanoseconds
#  gctimes - garbage collector times in nanoseconds
#  memory  - memory allocation in bytes
#  allocs  - number of allocations

struct BenchmarkEntry
    group::String
    len::Vector{Int}
    allocs::Vector{Int}
    memory::Vector{Int}
    nops::Vector{Float64}
    min::Vector{Float64}
    max::Vector{Float64}
    avg::Vector{Float64}
    std::Vector{Float64}
    median::Vector{Float64}
    BenchmarkEntry(group::String) = new(group, Int[], Int[], Int[],
                                        Float64[], Float64[], Float64[],
                                        Float64[], Float64[], Float64[])
end

function push_result!(e::BenchmarkEntry,
                      len::Integer,
                      nops::Union{Real,Function},
                      trial::Trial)
    push!(e.len, len)
    push!(e.allocs, trial.allocs)
    push!(e.memory, trial.memory)
    if isa(nops, Real)
        push!(e.nops, nops)
    else
        push!(e.nops, nops(len))
    end
    push!(e.min, minimum(trial.times))
    push!(e.max, maximum(trial.times))
    push!(e.avg, mean(trial.times))
    push!(e.std, std(trial.times; corrected = true))
    push!(e.median, median(trial.times))
    return e
end

get_L(A::Cholesky) = LowerTriangular(convert(Matrix, A.L))
get_U(A::Cholesky) = UpperTriangular(convert(Matrix, A.U))

function save_benchmarks(filename::String,
                         database::AbstractDict{String,BenchmarkEntry})
    # Auxiliary function.
    @inline function write_field!(dst::HDF5.Group,
                                  src::BenchmarkEntry,
                                  ::Val{sym}) where {sym}
        dst[string(sym)] = getfield(src, sym)
        nothing
    end
    temp = filename*".tmp"
    h5open(temp, "w") do f
        for group in keys(database)
            e = database[group]
            g = create_group(f, group)
            write_field!(g, e, Val(:len))
            write_field!(g, e, Val(:allocs))
            write_field!(g, e, Val(:memory))
            write_field!(g, e, Val(:nops))
            write_field!(g, e, Val(:min))
            write_field!(g, e, Val(:max))
            write_field!(g, e, Val(:avg))
            write_field!(g, e, Val(:std))
            write_field!(g, e, Val(:median))
            close(g)
        end
    end
    mv(temp, filename; force=true)
    nothing
end

load_benchmarks(filename::String) =
    load_benchmarks(Dict{String,BenchmarkEntry}(), filename)

function load_benchmarks(database::AbstractDict{String,BenchmarkEntry},
                         filename::String)
    # Auxiliary function.
    @inline function read_field!(dst::BenchmarkEntry,
                                 src::HDF5.Group,
                                 ::Val{sym},
                                 I::Vector{Int}) where {sym}
        A = read(src, string(sym))
        B = resize!(getfield(dst, sym), length(A))
        for i in eachindex(B, I)
            B[i] = A[I[i]]
        end
        nothing
    end
    h5open(filename, "r") do f
        for type in keys(f)
            type_grp = f[type]
            for alg in keys(type_grp)
                alg_grp = type_grp[alg]
                for opt in keys(alg_grp)
                    g = alg_grp[opt]
                    len = read(g, "len")
                    i = sortperm(len)
                    e = BenchmarkEntry(type*"/"*alg*"/"*opt)
                    read_field!(e, g, Val(:len),    i)
                    read_field!(e, g, Val(:allocs), i)
                    read_field!(e, g, Val(:memory), i)
                    read_field!(e, g, Val(:nops),   i)
                    read_field!(e, g, Val(:min),    i)
                    read_field!(e, g, Val(:max),    i)
                    read_field!(e, g, Val(:avg),    i)
                    read_field!(e, g, Val(:std),    i)
                    read_field!(e, g, Val(:median), i)
                    database[e.group] = e
                end
            end
        end
    end
    return database
end

function run_benchmarks(filename::AbstractString,
                        alg::Union{AbstractString,Symbol}; kwds...)
    return run_benchmarks(filename, Symbol(alg); kwds...)
end

function run_benchmarks(filename::AbstractString,
                        alg::Symbol; kwds...)
    if alg === :ldiv
        return run_benchmarks(
            filename,
            map(x->LDiv{x}, (Standard,Debug,InBounds,Vectorize))...;
            kwds...)
    elseif alg === :Cholesky
        algs = DataType[]
        push!(algs, CholeskyBLAS)
        #push!(algs, CholeskyStatic)
        for A in (CholeskyBanachiewiczLowerII,
                  CholeskyBanachiewiczLowerI,
                  CholeskyBanachiewiczUpper,
                  CholeskyCroutLower,
                  CholeskyCroutUpperII,
                  CholeskyCroutUpperI)
            for opt in (Debug, InBounds, Vectorize)
                push!(algs, A{opt})
            end
        end
        return run_benchmarks(filename, algs...; kwds...)
    else
        error("unknown algorithm ", alg)
    end
end

function run_benchmarks(filename::AbstractString,
                        args::Type{<:AbstractAlgorithm}...;
                        sizes = 10:10:300,
                        types = (Float32,))
    db = Dict{String,BenchmarkEntry}()
    if isfile(filename)
        load_benchmarks(db, filename)
    end
    for T in types, n in sizes
        run_benchmarks(filename, db, T, Int(n), args...)
    end
    return db
end

function run_benchmarks(filename::AbstractString,
                        db::AbstractDict{String,BenchmarkEntry},
                        T::Type{<:Floats},
                        n::Int,
                        args::Type{<:AbstractAlgorithm}...)
    A = generate_positive_definite_matrix(T, n)
    C = cholesky(A)
    x = generate_array(T, n)
    for X in args
        run_benchmarks(filename, db, X, A, C, x)
    end
    return db
end

instances(X::Type{<:AbstractAlgorithm}) = (X(),)
instances(X::Type{<:LDiv}) = (X("L"), X("L'"), X("R"), X("R'"))

function run_benchmarks(filename::AbstractString,
                        db::AbstractDict{String,BenchmarkEntry},
                        X::Type{<:AbstractAlgorithm},
                        A::Matrix{T},
                        C::Cholesky{T},
                        x::Vector{T}) where {T<:Floats}
    n = length(x)
    for alg in instances(X)
        grp = group(T, alg)
        if haskey(db, grp)
            if n ∈ db[grp].len
                io = stderr
                printstyled(io, "warning:"; color=:yellow)
                print(io, " skipping test for ")
                printstyled(io, grp; color=:cyan)
                print(io, " with ")
                printstyled(io, "n=", n; color=:cyan)
                print(io, " element(s)\n")
                continue
            end
            e = db[grp]
        else
            e = BenchmarkEntry(grp)
            db[grp] = e
        end
        run_benchmark(e, alg, A, C, x)
        save_benchmarks(filename, db)
    end
    return db
end

function run_benchmark(e::BenchmarkEntry,
                       alg::LDiv{opt},
                       A::Matrix{T},
                       C::Cholesky{T},
                       x::Vector{T}) where {T<:Floats,opt}
    n = length(x)
    w = similar(x)
    if alg.operand == "L"
        L = get_L(C)
        push_result!(e, n, nops(alg, n),
                     @benchmark ldiv!($opt, $w, $(L),  $x))
    elseif alg.operand == "L'"
        L = get_L(C)
        push_result!(e, n, nops(alg, n),
                     @benchmark ldiv!($opt, $w, $(L'), $x))
    elseif alg.operand == "R"
        R = get_U(C)
        push_result!(e, n, nops(alg, n),
                     @benchmark ldiv!($opt, $w, $(R),  $x))
    elseif alg.operand == "R'"
        R = get_U(C)
        push_result!(e, n, nops(alg, n),
                     @benchmark ldiv!($opt, $w, $(R'), $x))
    end
    return e
end

function run_benchmark(e::BenchmarkEntry,
                       alg::CholeskyBLAS,
                       A::Matrix{T},
                       C::Cholesky{T},
                       x::Vector{T}) where {T<:Floats}
    n = length(x)
    B = similar(A)
    opt = Standard
    push_result!(e, n, nops(alg, n), @benchmark cholesky!($opt, $B, $A))
end

function run_benchmark(e::BenchmarkEntry,
                       alg::CholeskyStatic,
                       A::Matrix{T},
                       C::Cholesky{T},
                       x::Vector{T}) where {T<:Floats}
    run_benchmark(e, alg, A, C, x, Val(length(x)))
end

function run_benchmark(e::BenchmarkEntry,
                       alg::CholeskyStatic,
                       A::Matrix{T},
                       C::Cholesky{T},
                       x::Vector{T},
                       ::Val{n}) where {T<:Floats,n}
    Asm = SMatrix{n,n}(A)
    push_result!(e, n, nops(alg, n), @benchmark cholesky($Asm))
end

function run_benchmark(e::BenchmarkEntry,
                       alg::CholeskyFactorization,
                       A::Matrix{T},
                       C::Cholesky{T},
                       x::Vector{T}) where {T<:Floats}
    n = length(x)
    B = similar(A)
    push_result!(e, n, nops(alg, n), @benchmark cholesky!($alg,  $B, $A))
end

# Generate random values uniformly distributed in [-1,1].
generate_array(T::Type{<:Real}, dims...) = 2*rand(T, dims...) .- 1
generate_array(T::Type{<:Complex{<:Real}}, dims...) =
    2*rand(T, dims...) .- Complex(1,1)

# Build a random Hermitian positive definite matrix.
function generate_positive_definite_matrix(T::Type, n::Integer)
    H = generate_array(T, 10n, n)
    A = H'*H
    return (A + A')/2
end

#plt.figure(2); plt.clf();for lab in ("Standard","Debug","InBounds","Vectorize"); dat = q["ldiv!,$lab,L'"];plt.plot(dat.len, dat.nops ./ dat.median, label=lab); end; plt.legend();plt.xlabel("size");plt.ylabel("Gflops (median)");plt.title("Left division by transpose of lower triangular matrix");savefig("ldiv-Lt-median.png",bbox_inches="tight")

#figure(2); clf(2); db = BenchmarkingMayOptimizeLinearAlgebraMethods.load_benchmarks("benchmarks-1.h5");clf(); for o in ("BLAS","Debug","InBounds","Vectorize"); g="Float32/ldiv!-R/$o"; plot(db[g].len, db[g].nops./db[g].median,label=g); end; legend();

# Cholesky:
# BLAS is:
# 11.3 Gflops for Float64 for n=200
# 10.7 Gflops for Float32 for n=200

# CholeskyCroutUpperII
# 8.77 Gflops for Float64 for n=200
# 11.3 Gflops for Float32 for n=200

# CholeskyBanachiewiczUpper (this is the fastest):
# 9.34 Gflops for Float64 for n=200
# 13.0 Gflops for Float32 for n=200

make_plot(filename::AbstractString, args...; kwds...) =
     make_plot(load_benchmarks(filename), args...; kwds...)

function make_plot(db::AbstractDict{String,BenchmarkEntry}, plt;
                   type = "Float32", op = :ldiv, field = :median,
                   first_figure = 1,
                   save::Bool = false, dir = ".", ext = ".png",
                   dpi = 120, pad_inches = 5/dpi)

    fig = first_figure - 1
    if op === :ldiv
        for A in ("L", "L'", "R", "R'")
            fig += 1
            plt.figure(fig)
            plt.clf()
            plt.tight_layout()
            for o in ("BLAS", "Debug", "InBounds", "Vectorize")
                g = "$type/ldiv!-$A/$o"
                plt.plot(db[g].len, db[g].nops./getfield(db[g], field), label=o);
            end
            plt.legend()
            plt.title(string("Left division by ",
                             (endswith(A, "'") ? "transpose of " : ""),
                             (endswith(A, r"L'?") ? "lower" : "upper"),
                             " triangular matrix"))
            if save
                filename = joinpath(
                    dir, string("ldiv-", replace(A, r"'$" => "t"), "-",
                                field, ext))
                plt.savefig(filename; dpi=dpi, pad_inches=pad_inches)
            end
        end
    elseif op === :cholesky
        for opt in (:Debug, :InBounds, :Vectorize)
            fig += 1
            plt.figure(fig)
            plt.clf()
            plt.tight_layout()
            for f in (:BLAS,
                      :CholeskyBanachiewiczLowerI,
                      :CholeskyBanachiewiczLowerII,
                      :CholeskyBanachiewiczUpper,
                      :CholeskyCroutLower,
                      :CholeskyCroutUpperI,
                      :CholeskyCroutUpperII)
                g = string(type, (f === :BLAS ? "/Cholesky/$f" : "/$f/$opt"))
                plt.plot(db[g].len, db[g].nops./getfield(db[g], field), label="$f");
            end
            plt.legend()
            plt.title("Cholesky decomposition (optimization level: `$opt`)")
            if save
                filename = joinpath(
                    dir, string("cholesky-", lowercase(string(opt)), "-",
                                field, ext))
                plt.savefig(filename; dpi=dpi, pad_inches=pad_inches)
            end
        end
    else
        error("unknown operation: ", op)
    end
end

end # module
