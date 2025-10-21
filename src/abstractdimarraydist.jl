# Required interface for all AbstractDimArrayDistribution subtypes
# - methods required for array-variate distributions in Distributions.jl, especially:
#   + Random.rand!(rng, ::AbstractDimArrayDistribution{N}, x::AbstractArray{<:Real,N})
#   + Distributions.logpdf(dist::AbstractDimArrayDistribution{N}, x::AbstractArray{<:Real,N})
# AND
# - Dimensions.dims(dist)
# - DimensionalData.rebuild(dist; dims, kw...)
abstract type AbstractDimArrayDistribution{N,Supp<:Distributions.ValueSupport} <:
              ArrayDistribution{N,Supp} end

function Base.axes(dist::AbstractDimArrayDistribution)
    dims = Dimensions.dims(dist)
    ax = map(axes ∘ DimensionalData.lookup, dims)
    return map(Dimensions.DimUnitRange, ax, dims)
end

function Base.rand(rng::Random.AbstractRNG, dist::AbstractDimArrayDistribution)
    x = DimensionalData.DimArray{_eltype(dist)}(undef, _dims(dist))
    Random.rand!(rng, Distributions.sampler(dist), parent(x))
    return x
end
function Base.rand(
    rng::Random.AbstractRNG, dist::AbstractDimArrayDistribution, lengths::Dims
)
    dims = Dimensions.dims(dist)
    ax = map(Base.OneTo, lengths)
    out = [
        DimensionalData.DimArray{_eltype(dist)}(undef, dims) for
        _ in Iterators.product(ax...)
    ]
    return Random.rand!(rng, Distributions.sampler(dist), out, false)
end
function Base.rand(
    rng::Random.AbstractRNG, dist::AbstractDimArrayDistribution{0}, lengths::Dims
)
    out = Array{_eltype(dist)}(undef, lengths)
    Random.rand!(rng, Distributions.sampler(dist), out)
    return out
end
