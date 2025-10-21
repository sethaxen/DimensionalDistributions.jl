_eltype(dist::AbstractDimArrayDistribution) = eltype(dist)
function _eltype(dist::AbstractDimArrayDistribution{<:Any,<:Distributions.Continuous})
    return float(eltype(dist))
end

_astuple(x::Tuple) = x
_astuple(::Nothing) = ()

_dims(x) = _astuple(Dimensions.dims(x))

_maybe_dimarray(x::AbstractArray, dims::Tuple) = DimensionalData.DimArray(x, dims)
_maybe_dimarray(x, ::Tuple) = x
