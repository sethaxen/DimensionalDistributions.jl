module DimensionalDistributions

using DimensionalData: DimensionalData, Dimensions
using Distributions: Distributions
using Random: Random
using Statistics: Statistics
using StatsBase: StatsBase

export withdims

const ArrayDistribution{N,S<:Distributions.ValueSupport} = Distributions.Distribution{
    Distributions.ArrayLikeVariate{N},S
}

include("abstractdimarraydist.jl")
include("asdimarraydist.jl")
include("withdims.jl")
include("utils.jl")

end  # module
