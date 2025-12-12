using Bijectors
using DimensionalDistributions
using Test

@testset "Bijectors integration" begin
    all_dims = (X(1:2), Y(1:2), Z(1:2))
    @testset for udist in [Normal(), Beta()], ndims in eachindex(all_dims)
        dim = all_dims[1:ndims]
        prod_dist = product_distribution(fill(udist, size(dim)))
        dim_dist = DimensionalDistributions.AsDimArrayDistribution(prod_dist, dim)
        @test Bijectors.bijector(dim_dist) isa typeof(Bijectors.bijector(prod_dist))
    end
end
