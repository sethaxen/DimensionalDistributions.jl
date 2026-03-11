using Bijectors
using DimensionalData
using DimensionalDistributions
using Distributions
using Random

struct TestDimDistribution{M<:UnivariateDistribution,N,Dims<:Tuple,S<:ValueSupport} <:
       DimensionalDistributions.AbstractDimArrayDistribution{N,S}
    dist::M
    dims::Dims
    function TestDimDistribution(
        dist::M, dims::Dims
    ) where {M<:UnivariateDistribution,Dims<:Tuple}
        S = Distributions.value_support(M)
        N = length(dims)
        return new{M,N,Dims,S}(dist, dims)
    end
end

Dimensions.dims(dist::TestDimDistribution) = dist.dims
function DimensionalData.rebuild(
    dist::TestDimDistribution; dims=Dimensions.dims(dist), kw...
)
    return TestDimDistribution(dist.dist, dims)
end

Base.size(dist::TestDimDistribution) = map(length, dims(dist))
Base.length(dist::TestDimDistribution) = prod(size(dist))
Base.eltype(dist::TestDimDistribution) = eltype(dist.dist)
Base.eltype(::Type{<:TestDimDistribution{M}}) where {M} = eltype(M)
Distributions.params(dist::TestDimDistribution) = params(dist.dist)
Distributions.partype(dist::TestDimDistribution) = partype(dist.dist)

function Distributions.logpdf(
    dist::TestDimDistribution{<:Any,N}, x::AbstractArray{<:Real,N}
) where {N}
    @assert size(x) == size(dist)
    if x isa AbstractDimArray
        Dimensions.comparedims(dist, x)
    end
    return sum(xi -> logpdf(dist.dist, xi), x)
end
function Distributions.logpdf(dist::TestDimDistribution{<:Any,0}, x::Real)
    return logpdf(dist.dist, x)
end

function Random.rand!(
    rng::Random.AbstractRNG, dist::TestDimDistribution{<:Any,N}, x::AbstractArray{<:Real,N}
) where {N}
    rand!(rng, dist.dist, x)
    return x
end
function Base.rand(rng::Random.AbstractRNG, dist::TestDimDistribution{<:Any,0})
    return rand(rng, dist.dist)
end

function Bijectors.bijector(dist::TestDimDistribution{<:ContinuousUnivariateDistribution})
    return Bijectors.elementwise(Bijectors.bijector(dist.dist))
end
