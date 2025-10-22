# DimensionalDistributions

[![Build Status](https://github.com/sethaxen/DimensionalDistributions.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/sethaxen/DimensionalDistributions.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/sethaxen/DimensionalDistributions.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/sethaxen/DimensionalDistributions.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/invenia/BlueStyle)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![Aqua](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl)

This package provides useful integration between Distributions.jl and DimensionalData.jl.
Its API is a single function `withdims(::Distribution, dims)`, which one can use to decorate an array-variate distribution with dimensions.

## Example: tracking dimensions in a Turing model

In this example we build the 8-schools model in its non-centered parameterization.
All inputs and parameters with the school dimension are `DimArray`s.
We then decorate all appropriate distributions with the same dimension using `withdim`.
For the `DimArray`s to be preserved in the object returned by `sample`, we need to use FlexiChains.jl.

```julia
using DimensionalData, DimensionalDistributions, LinearAlgebra, PDMats, Turing, FlexiChains, Random
using FlexiChains: VNChain

Random.seed!(42)

schools = [
    "Choate",
    "Deerfield",
    "Phillips Andover",
    "Phillips Exeter",
    "Hotchkiss",
    "Lawrenceville",
    "St. Paul's",
    "Mt. Hermon",
]

school_dim = Dim{:school}(schools)
y = DimArray([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0], school_dim)
σ = DimArray([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0], school_dim)

@model function noncentered_eight(σ; dim=only(dims(σ)), n=length(σ))
    μ ~ Normal(0, 5)
    τ ~ truncated(Cauchy(0, 5); lower=0)
    θ_tilde ~ withdims(filldist(Normal(), n), dim)
    θ := @. μ + τ * θ_tilde
    y ~ withdims(arraydist(Normal.(θ, σ)), dim)
end

model = noncentered_eight(σ)
model_cond = model | (; y)
chns_prior = sample(model, Prior(), 4_000; chain_type=VNChain)
chns = sample(model_cond, NUTS(), MCMCThreads(), 1_000, 4; chain_type=VNChain)
```

Within `chns` and `chns_prior`, `θ_tilde` and `θ` are stored as `DimArray`s.
```julia
julia> mean(chns[@varname(θ)])
┌ 8-element DimArray{Float64, 1} ┐
├────────────────────────────────┴─────────────────────────────── dims ┐
  ↓ school Categorical{String} ["Choate", …, "Mt. Hermon"] Unordered
└──────────────────────────────────────────────────────────────────────┘
 "Choate"            6.22847
 "Deerfield"         4.90917
 "Phillips Andover"  3.83916
 "Phillips Exeter"   4.67192
 "Hotchkiss"         3.59698
 "Lawrenceville"     4.00371
 "St. Paul's"        6.17751
 "Mt. Hermon"        4.80628
```
