using DimensionalData
using DimensionalDistributions
using Distributions
using Random
using StatsBase
using Test

@testset "AsDimArrayDistribution" begin
    distributions = [
        Normal(randn(), rand()),
        Beta(1.0f0, 2.0f0),
        Bernoulli(0.25),
        MvNormal(randn(5), rand(5)),
        Dirichlet(rand(5) .+ 1),
        Multinomial(10, rand(Dirichlet(ones(5)))),
        LKJ(4, 2.0),
        product_distribution(fill(Normal(randn(), rand()), 2, 3, 4)),
    ]
    dims = (X, Y, Z, Ti)
    @testset for dist in distributions
        ndims = length(size(dist))
        dim_types = Tuple(sample(collect(dims), ndims; replace=false))
        dim = ntuple(i -> dim_types[i](1:size(dist)[i]), ndims)
        dim_dist = DimensionalDistributions.AsDimArrayDistribution(dist, dim)
        dim_alt = if ndims == 1
            (first(setdiff(dims, dim_types))(1:length(dist)),)
        else
            ntuple(i -> dim_types[end - i + 1](1:size(dist)[i]), ndims)
        end

        @testset "constructor/accessors" begin
            @inferred DimensionalDistributions.AsDimArrayDistribution(dist, dim)
            dim_nt = NamedTuple{Dimensions.name(dim_types)}(
                ntuple(i -> (1:size(dist)[i]), ndims)
            )
            @testset for _dim in (dim, dim_types, dim_nt)
                _dim_dist = DimensionalDistributions.AsDimArrayDistribution(dist, _dim)
                @test _dim_dist isa DimensionalDistributions.AsDimArrayDistribution
                @test parent(_dim_dist) === dist
                @test Dimensions.dims(_dim_dist) isa Tuple{Vararg{<:Dimensions.Dimension}}
                @test Dimensions.comparedims(Bool, Dimensions.dims(_dim_dist), dim)
            end
        end

        @testset "rebuild" begin
            @test rebuild(dim_dist) === dim_dist
            dim_dist_alt = rebuild(dim_dist; dims=dim_alt, name=:foo, metatdata=())
            @test dim_dist_alt isa DimensionalDistributions.AsDimArrayDistribution
            @test parent(dim_dist_alt) === dist
            @test Dimensions.comparedims(Bool, Dimensions.dims(dim_dist_alt), dim_alt)
        end

        @testset "Distributions interface consistent with parent" begin
            @test axes(dim_dist) == axes(dist)
            @test eltype(dim_dist) === eltype(dist)
            @test eltype(typeof(dim_dist)) === eltype(typeof(dist))
            @test partype(dim_dist) === partype(dist)
            @test params(dim_dist) == params(dist)
            @test size(dim_dist) == size(dist)
            @test length(dim_dist) == length(dist)

            @testset "support" begin
                if ndims == 0
                    @test support(dim_dist) === support(dist)
                end
                x_in_support = rand(dist)
                x_notin_support = broadcast(_ -> NaN, x_in_support)
                @test insupport(dim_dist, x_in_support) == insupport(dist, x_in_support)
                @test insupport(dim_dist, x_notin_support) ==
                    insupport(dist, x_notin_support)
                if ndims <= 1
                    x_stack = hcat(x_in_support, x_notin_support)
                    @test insupport(dim_dist, x_stack) == insupport(dist, x_stack)
                end
                @testset for f in (minimum, maximum, extrema)
                    f_dist = try
                        f(dist)
                    catch
                        continue
                    end
                    @test f(dim_dist) == f_dist
                end
            end

            @testset "rand" begin
                @testset for sz in ((), (3,), (3, 4), (3, 4, 5))
                    @testset for seed in [42, 98]
                        x = rand(MersenneTwister(seed), dist, sz...)
                        x_dim = rand(MersenneTwister(seed), dim_dist, sz...)
                        @test x == x_dim
                    end
                end
            end

            @testset "logpdf/pdf/loglikelihood" begin
                @testset for sz in ((), (3,), (3, 4), (3, 4, 5))
                    x = rand(dist, sz...)
                    @test logpdf(dim_dist, x) == logpdf(dist, x)
                    @test pdf(dim_dist, x) == pdf(dist, x)
                    @test loglikelihood(dim_dist, x) == loglikelihood(dist, x)
                end
            end

            @testset "statistical properties" begin
                # only test those that are defined for the parent distribution
                @testset for f in (mean, var, std, mode, cor, cov, invcov)
                    f_dist = try
                        f(dist)
                    catch
                        continue
                    end
                    @test f(dim_dist) == f_dist
                end
            end
        end

        ndims > 0 && @testset "dimensions validated when appropriate" begin
            x = DimArray(rand(dist), dim_alt)
            @testset "insupport" begin
                @test insupport(dim_dist, Array(x))
                @test !insupport(dim_dist, x)
            end

            @testset for f in (logpdf, pdf, loglikelihood)
                @test f(dim_dist, Array(x)) isa Real
                @test_throws DimensionMismatch f(dim_dist, x)
            end
        end

        @testset "dimensional arrays returned when appropriate" begin
            @testset "rand" begin
                x = rand(dim_dist)
                if ndims == 0
                    @test x isa Real
                else
                    @test x isa DimArray
                    @test Dimensions.comparedims(Bool, dim_dist, x)
                end
                @testset for sz in ((3,), (3, 4))
                    x = rand(dim_dist, sz...)
                    @test x isa Union{Array{<:Real},Array{<:DimArray}}
                end
                sample_dims = ntuple(i -> Dim{Symbol(:draw, i)}(1:i), 3)
                @testset for n_sample_dims in eachindex(sample_dims)
                    sample_dim = sample_dims[1:n_sample_dims]
                    x = rand(dim_dist, sample_dim)
                    @test x isa DimArray
                    if ndims == 0
                        @test x isa DimArray{<:Real}
                        @test Dimensions.comparedims(Bool, sample_dim, Dimensions.dims(x))
                    else
                        @test x isa DimArray{<:DimArray}
                        @test Dimensions.comparedims(Bool, sample_dim, Dimensions.dims(x))
                        @test Dimensions.comparedims(Bool, dim, Dimensions.dims(first(x)))
                    end
                end
            end
        end

        @testset "product_distribution" begin end
    end
end
