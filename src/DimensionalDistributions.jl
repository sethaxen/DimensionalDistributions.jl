module DimensionalDistributions

using DimensionalData: DimensionalData, Dimensions
using Distributions: Distributions
using Random: Random

const ArrayDistribution{N,S<:Distributions.ValueSupport} = Distributions.Distribution{
    Distributions.ArrayLikeVariate{N},S
}

include("abstractdimarraydist.jl")
include("utils.jl")

end  # module
