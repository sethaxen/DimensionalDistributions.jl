struct AsDimArrayDistribution{
    N,Dist<:ArrayDistribution{N},Dims<:Tuple,Supp<:Distributions.ValueSupport
} <: AbstractDimArrayDistribution{N,Supp}
    dist::Dist
    dims::Dims
    function AsDimArrayDistribution(dist::ArrayDistribution{N}, dims) where {N}
        _dims = Dimensions.format(dims, CartesianIndices(axes(dist)))
        supp = Distributions.value_support(typeof(dist))
        return new{N,typeof(dist),typeof(_dims),supp}(dist, _dims)
    end
end

Base.parent(dist::AsDimArrayDistribution) = getfield(dist, :dist)

function Base.axes(dist::AsDimArrayDistribution)
    return map(Dimensions.DimUnitRange, axes(parent(dist)), Dimensions.dims(dist))
end

# DimensionalData interface methods

Dimensions.dims(dist::AsDimArrayDistribution) = getfield(dist, :dims)

function DimensionalData.rebuild(
    dist::AsDimArrayDistribution; dims=Dimensions.dims(dist), kw...
)
    return AsDimArrayDistribution(parent(dist), dims)
end

# Distributions interface methods

Distributions.partype(dist::AsDimArrayDistribution) = Distributions.partype(parent(dist))
Distributions.params(dist::AsDimArrayDistribution) = Distributions.params(parent(dist))

Base.size(dist::AsDimArrayDistribution) = size(parent(dist))
Base.length(dist::AsDimArrayDistribution) = length(parent(dist))
Base.eltype(dist::AsDimArrayDistribution) = eltype(parent(dist))
Base.eltype(::Type{<:AsDimArrayDistribution{N,Dist}}) where {N,Dist} = eltype(Dist)

Distributions.support(dist::AsDimArrayDistribution{0}) = Distributions.support(parent(dist))
function Distributions.insupport(
    dist::AsDimArrayDistribution{N}, x::AbstractArray{<:Real,N}
) where {N}
    return Distributions.insupport(parent(dist), x)
end
function Distributions.insupport(
    dist::AsDimArrayDistribution{N}, x::DimensionalData.AbstractDimArray{<:Real,N}
) where {N}
    return Dimensions.comparedims(Bool, dist, x) && Distributions.insupport(dist, parent(x))
end
function Distributions.insupport(dist::AsDimArrayDistribution{0}, x::Real)
    return Distributions.insupport(parent(dist), x)
end
function Distributions.insupport(dist::AsDimArrayDistribution{1}, x::AbstractMatrix{<:Real})
    out = similar(BitVector, axes(x)[2])
    map!(Base.Fix1(Distributions.insupport, dist), out, eachcol(x))
    return out
end

for _fname in (:minimum, :maximum)
    @eval function Base.$(_fname)(dist::AsDimArrayDistribution)
        return _maybe_dimarray($(_fname)(parent(dist)), Dimensions.dims(dist))
    end
end
function Base.extrema(dist::AsDimArrayDistribution)
    extrema = Base.extrema(parent(dist))
    return map(Base.Fix2(_maybe_dimarray, Dimensions.dims(dist)), extrema)
end

for _fname in (:logpdf, :pdf)
    @eval function Distributions.$(_fname)(
        dist::AsDimArrayDistribution{N}, x::DimensionalData.AbstractArray{<:Real,N}
    ) where {N}
        if x isa DimensionalData.AbstractDimArray
            Dimensions.comparedims(dist, x)
        end
        return Distributions.$(_fname)(parent(dist), x)
    end
    @eval function Distributions.$(_fname)(
        dist::AsDimArrayDistribution{0}, x::DimensionalData.AbstractArray{<:Real,0}
    )
        return Distributions.$(_fname)(parent(dist), x)
    end
    @eval function Distributions.$(_fname)(dist::AsDimArrayDistribution{0}, x::Real)
        return Distributions.$(_fname)(parent(dist), x)
    end
end

for (_module, _fname) in
    ((:Statistics, :mean), (:Statistics, :var), (:Statistics, :std), (:StatsBase, :mode))
    @eval function $(_module).$(_fname)(dist::AsDimArrayDistribution{N}) where {N}
        return _maybe_dimarray($(_module).$(_fname)(parent(dist)), Dimensions.dims(dist))
    end
end
for (_module, _fname) in
    ((:Statistics, :cov), (:Statistics, :cor), (:Distributions, :invcov))
    @eval function $(_module).$(_fname)(dist::AsDimArrayDistribution{1})
        dim = only(Dimensions.dims(dist))
        return DimensionalData.DimArray($(_module).$(_fname)(parent(dist)), (dim, dim))
    end
end

function Distributions.product_distribution(
    dists::DimensionalData.AbstractDimArray{<:AsDimArrayDistribution}
)
    dims1 = Dimensions.dims(first(dists))
    all(@view(dists[(begin + 1):end])) do dist
        Dimensions.comparedims(Bool, dims1, Dimensions.dims(dist))
    end || return Distributions.ProductDistribution(map(parent, dists))
    prod_dist = Distributions.product_distribution(map(parent, dists))
    prod_dims = (dims1..., Dimensions.dims(dists)...)
    return withdims(prod_dist, prod_dims)
end

Distributions.sampler(dist::AsDimArrayDistribution) = Distributions.sampler(parent(dist))

function Random.rand!(
    rng::Random.AbstractRNG, dist::AsDimArrayDistribution{N}, x::AbstractArray{<:Real,N}
) where {N}
    return Random.rand!(rng, parent(dist), x)
end
function Base.rand(rng::Random.AbstractRNG, dist::AsDimArrayDistribution{0})
    return Base.rand(rng, parent(dist))
end
function Base.rand(rng::Random.AbstractRNG, dist::AsDimArrayDistribution{0}, lengths::Dims)
    return Base.rand(rng, parent(dist), lengths)
end
