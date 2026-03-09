module DimensionalDistributionsBijectorsExt

using DimensionalDistributions: DimensionalDistributions
using Bijectors: Bijectors

function Bijectors.bijector(dist::DimensionalDistributions.AsDimArrayDistribution)
    return Bijectors.bijector(parent(dist))
end

_zero(::AbstractArray{T}) where {T<:Number} = zero(real(T))
_zero(::AbstractArray) = 0.0

@static if isdefined(Bijectors, :VectorBijectors)
    using DimensionalData: DimensionalData, Dimensions
    using Bijectors: VectorBijectors

    # --- Private helper types ---

    # Wraps a plain array into a DimArray with given dims (zero ladj)
    struct WrapDimArray{D}
        dims::D
    end
    (f::WrapDimArray)(x) = DimensionalData.DimArray(x, f.dims)
    Bijectors.inverse(f::WrapDimArray) = UnwrapDimArray(f.dims)
    Bijectors.with_logabsdet_jacobian(f::WrapDimArray, x) = (f(x), _zero(x))

    # Strips DimArray wrapper, returning parent array (zero ladj)
    struct UnwrapDimArray{D}
        dims::D
    end
    (f::UnwrapDimArray)(x::DimensionalData.AbstractDimArray) = parent(x)
    (f::UnwrapDimArray)(x) = x
    Bijectors.inverse(f::UnwrapDimArray) = WrapDimArray(f.dims)
    function Bijectors.with_logabsdet_jacobian(f::UnwrapDimArray, x::AbstractArray)
        return (f(x), _zero(x))
    end

    # Flattens an array to a vector (adapted from Bijectors.jl)
    struct Vec{N}
        size::NTuple{N,Int}
    end
    (::Vec)(x::AbstractArray) = vec(x)
    Bijectors.inverse(v::Vec) = Reshape(v.size)
    Bijectors.with_logabsdet_jacobian(::Vec, x::AbstractArray) = (vec(x), _zero(x))

    # Reshapes a vector to an array (adapted from Bijectors.jl)
    struct Reshape{N}
        size::NTuple{N,Int}
    end
    (r::Reshape)(x::AbstractArray) = reshape(x, r.size)
    Bijectors.inverse(r::Reshape) = Vec(r.size)
    Bijectors.with_logabsdet_jacobian(r::Reshape, x::AbstractArray) = (r(x), _zero(x))

    # --- VectorBijectors interface for AbstractDimArrayDistribution ---

    function VectorBijectors.from_vec(
        d::DimensionalDistributions.AbstractDimArrayDistribution
    )
        return WrapDimArray(Dimensions.dims(d)) ∘ Reshape(size(d))
    end
    function VectorBijectors.to_vec(
        d::DimensionalDistributions.AbstractDimArrayDistribution
    )
        return Vec(size(d)) ∘ UnwrapDimArray(Dimensions.dims(d))
    end
    function VectorBijectors.from_linked_vec(
        d::DimensionalDistributions.AbstractDimArrayDistribution
    )
        b = Bijectors.bijector(d)
        sz = Bijectors.output_size(b, size(d))
        return WrapDimArray(Dimensions.dims(d)) ∘ Bijectors.inverse(b) ∘ Reshape(sz)
    end
    function VectorBijectors.to_linked_vec(
        d::DimensionalDistributions.AbstractDimArrayDistribution
    )
        b = Bijectors.bijector(d)
        sz = Bijectors.output_size(b, size(d))
        return Vec(sz) ∘ b ∘ UnwrapDimArray(Dimensions.dims(d))
    end
    function VectorBijectors.linked_vec_length(
        d::DimensionalDistributions.AbstractDimArrayDistribution
    )
        return prod(Bijectors.output_size(Bijectors.bijector(d), size(d)))
    end

    # --- VectorBijectors interface for AsDimArrayDistribution (dispatch to parent) ---

    function VectorBijectors.from_vec(d::DimensionalDistributions.AsDimArrayDistribution)
        return WrapDimArray(Dimensions.dims(d)) ∘ VectorBijectors.from_vec(parent(d))
    end
    function VectorBijectors.to_vec(d::DimensionalDistributions.AsDimArrayDistribution)
        return VectorBijectors.to_vec(parent(d)) ∘ UnwrapDimArray(Dimensions.dims(d))
    end
    function VectorBijectors.from_linked_vec(
        d::DimensionalDistributions.AsDimArrayDistribution
    )
        return WrapDimArray(Dimensions.dims(d)) ∘ VectorBijectors.from_linked_vec(parent(d))
    end
    function VectorBijectors.to_linked_vec(
        d::DimensionalDistributions.AsDimArrayDistribution
    )
        return VectorBijectors.to_linked_vec(parent(d)) ∘ UnwrapDimArray(Dimensions.dims(d))
    end
    function VectorBijectors.linked_vec_length(
        d::DimensionalDistributions.AsDimArrayDistribution
    )
        return VectorBijectors.linked_vec_length(parent(d))
    end
end

end  # module
