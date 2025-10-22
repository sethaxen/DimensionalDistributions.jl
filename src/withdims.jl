
"""
    withdims(dist::Distribution{<:ArrayLikeVariate}, dims)

Decorate the array-variate distribution with DimensionalData dimensions (if any).

The result is an array-variate distribution `dim_dist` with the same support as the
original distribution but with additional functionality:

  - `Dimensions.dims(dim_dist)` returns the dimensions of the distribution.
  - `axes(dim_dist)` returns dimensional axes (if any)
  - `rand([rng,]dim_dist)` returns a random draw as a `DimArray` (if `dist` is not
    univariate)
  - `logpdf(dim_dist, x::AbstractDimArray)` and similar methods for `pdf` and
    `loglikelihood` statically validate the types of the dimensions of `x` against the
    dimensions of `dim_dist` before returning the result.
  - `insupport(dim_dist, x::AbstractDimArray)` will return `false` if the dimensions of `x`
    do not match the dimensions of `dim_dist`.
  - When appropriate and available, `mean`, `var`, `std`, `mode`, `cov`, `cor`, `invcov`,
    `minimum`, `maximum`, and `extrema` will return a `DimArray` with the same dimensions as
    the distribution.

# Extended help

## Examples

Here we decorate a multivariate normal distribution with dimensions `X`:

```jldoctest withdims
julia> using DimensionalData, Distributions, DimensionalDistributions, Random

julia> d = withdims(MvNormal(ones(3)), X);

julia> Dimensions.dims(d)
(↓ X)

julia> axes(d)
(X(Base.OneTo(3)),)
```

Basic statistics methods now return `DimArray`s with the same dimensions as the
distribution:

```jldoctest withdims
julia> mean(d)
┌ 3-element DimArray{Float64, 1} ┐
├────────────────────────── dims ┤
  ↓ X
└────────────────────────────────┘
 0.0
 0.0
 0.0

julia> cov(d)
┌ 3×3 DimArray{Float64, 2} ┐
├──────────────────── dims ┤
  ↓ X, → X
└──────────────────────────┘
 1.0  0.0  0.0
 0.0  1.0  0.0
 0.0  0.0  1.0
```

Dimensions are validated when evaluating the (log-)pdf:

```jldoctest withdims
julia> z = DimArray{Float64}(undef, Z(1:3));

julia> rand!(d, z); # when sampling into an existing array, dimensions aren't validated

julia> logpdf(d, z);
ERROR: DimensionMismatch: X and Z dims on the same axis.
...
```

Finally, if we draw multiple samples by providing a dimension with a size, the result is a
`DimArray` of `DimArray`s, which can be stacked to produce a single `DimArray`:

```jldoctest withdims
julia> stack(rand(d, Z(2)))
┌ 3×2 DimArray{Float64, 2} ┐
├──────────────────── dims ┤
  ↓ X, → Z
└──────────────────────────┘
 1.38512   0.820114
 0.67123   0.238159
 0.901136  0.324754
```
"""
withdims(dist::ArrayDistribution, dims=_dims(dist)) = AsDimArrayDistribution(dist, dims)
function withdims(dist::AbstractDimArrayDistribution, dims=_dims(dist))
    DimensionalData.rebuild(dist; dims)
end
