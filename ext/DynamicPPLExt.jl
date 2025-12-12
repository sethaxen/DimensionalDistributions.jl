module DynamicPPLExt

using Bijectors: Bijectors
using DimensionalData: DimensionalData, Dimensions
using DimensionalDistributions: DimensionalDistributions
using DynamicPPL: DynamicPPL

const ZeroLogProb = zero(float(Real))

struct DimArrayTransform{D} <: Bijectors.Bijector
    dims::D
end

(f::DimArrayTransform)(x) = DimensionalData.DimArray(x, f.dims)

(::Bijectors.Inverse{<:DimArrayTransform})(x) = x
Bijectors.with_logabsdet_jacobian(f::DimArrayTransform, x) = (f(x), ZeroLogProb)
function Bijectors.with_logabsdet_jacobian(::Bijectors.Inverse{<:DimArrayTransform}, x)
    return (x, ZeroLogProb)
end

function DynamicPPL.from_vec_transform(
    d::DimensionalDistributions.AbstractDimArrayDistribution
)
    reshape_transform = DynamicPPL.from_vec_transform_for_size(size(d))
    dim_array_transform = DimArrayTransform(Dimensions.dims(d))
    return dim_array_transform ∘ reshape_transform
end
function DynamicPPL.from_vec_transform(d::DimensionalDistributions.AsDimArrayDistribution)
    return DimArrayTransform(Dimensions.dims(d)) ∘ DynamicPPL.from_vec_transform(parent(d))
end

function DynamicPPL.from_linked_vec_transform(
    d::DimensionalDistributions.AbstractDimArrayDistribution
)
    f_link = Bijectors.bijector(d)
    reshape_transform = DynamicPPL.from_vec_transform(f_link, size(d))
    dim_array_transform = DimArrayTransform(Dimensions.dims(d))
    return dim_array_transform ∘ Bijectors.inverse(f_link) ∘ reshape_transform
end
function DynamicPPL.from_linked_vec_transform(
    d::DimensionalDistributions.AsDimArrayDistribution
)
    return DimArrayTransform(Dimensions.dims(d)) ∘
           DynamicPPL.from_linked_vec_transform(parent(d))
end

end  # module
