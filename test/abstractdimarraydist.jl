using DimensionalData
using DimensionalDistributions
using Distributions
using Random
using Statistics
using Test

@testset "AbstractDimArrayDistribution" begin
    all_dims = (X(1:2), Y(1:2), Z(1:2))
    dists = [Normal(randn(), rand()), Beta(1.0f0, 2.0f0), DiscreteUniform(1, 10)]
    rng = Random.default_rng()
    @testset for udist in dists, ndims in 0:length(all_dims)
        dim = all_dims[1:ndims]
        dist = TestDimDistribution(udist, dim)
        @test dist isa DimensionalDistributions.AbstractDimArrayDistribution
        if ndims > 0
            prod_dist = product_distribution(fill(udist, size(dist)))
        else
            prod_dist = udist
        end

        # first test our interface methods
        @test dims(dist) == dim
        @test rebuild(dist; dims=dim) == dist

        # now Distributions interface methods
        @test eltype(dist) == eltype(typeof(dist)) == eltype(udist)
        @test params(dist) == params(udist)
        @test partype(dist) == partype(udist)
        x1 = zeros(eltype(dist), dim)
        x2 = copy(Array(x1))
        rand!(MersenneTwister(42), dist, x1)
        rand!(MersenneTwister(42), dist, x2)
        @test x1 == x2

        # now test generics that should just work™
        @testset "rand" begin
            if ndims == 0
                @test rand(dist) isa Real
            else
                @test rand(dist) isa AbstractDimArray
                @test Dimensions.comparedims(Bool, dist, rand(dist))
            end
            x = rand(rng, dist, 3)
            if ndims <= 1
                @test x isa Array{<:Real,ndims+1}
                @test size(x) == (size(dist)..., 3)
            else
                @test x isa Vector{<:DimArray}
                @test length(x) == 3
                @test Dimensions.comparedims(Bool, dist, first(x))
            end
            Random.rand!(rng, dist, x)
            @testset for sz in ((3, 4), (3, 4, 5))
                x = rand(rng, dist, sz)
                @test size(x) == sz
                if ndims == 0
                    @test x isa Array{<:Real,length(sz)}
                else
                    @test x isa Array{<:DimArray,length(sz)}
                    @test Dimensions.comparedims(Bool, dist, first(x))
                end
                Random.rand!(rng, dist, x)
            end
            sample_dims = ntuple(i -> Dim{Symbol(:draw, i)}(1:i), 3)
            @testset for n_sample_dims in eachindex(sample_dims)
                sample_dim = sample_dims[1:n_sample_dims]
                x = rand(rng, dist, sample_dim)
                @test size(x) == size(sample_dim)
                if ndims == 0
                    @test x isa DimArray{<:Real,n_sample_dims}
                else
                    @test x isa DimArray{<:DimArray,n_sample_dims}
                    @test Dimensions.comparedims(Bool, sample_dim, dims(x))
                    @test Dimensions.comparedims(Bool, dist, first(x))
                end
                Random.rand!(rng, dist, x)
            end
        end

        @testset for f in (logpdf, pdf)
            @testset for sz in ((), (3,), (3, 4), (3, 4, 5))
                x = rand(rng, dist, sz...)
                if ndims == 0 && !isempty(sz)
                    f_result = @test_deprecated f(dist, x)
                    f_prod_result = @test_deprecated f(prod_dist, x)
                else
                    f_result = f(dist, x)
                    f_prod_result = f(prod_dist, x)
                end
                if !isempty(sz)
                    @test f_result isa Array{<:Real,length(sz)}
                else
                    @test f_result isa Real
                end
                @test f_result ≈ f_prod_result
                @test size(f_result) == sz
            end

            @testset "dimensional arrays returned when appropriate" begin
                sample_dim = ntuple(i -> Dim{Symbol(:draw, i)}(1:i), 3)
                x_dim = rand(rng, dist, sample_dim)
                if ndims == 0
                    f_dim_result = @test_deprecated f(dist, x_dim)
                else
                    f_dim_result = f(dist, x_dim)
                end
                @test f_dim_result isa DimArray{<:Real,3}
                @test Dimensions.comparedims(Bool, sample_dim, dims(f_dim_result))
            end
        end
    end
end
