module DimensionalDistributionsBijectorsExt

using DimensionalDistributions: DimensionalDistributions
using Bijectors: Bijectors

function Bijectors.bijector(dist::DimensionalDistributions.AsDimArrayDistribution)
    return Bijectors.bijector(parent(dist))
end

end  # module
