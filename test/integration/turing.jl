using DimensionalData
using DimensionalDistributions
using FlexiChains
using Turing
using Test

function flexichains_sample_dims(chn::VNChain)
    return (
        DimensionalData.Dim{:iter}(FlexiChains.iter_indices(chn)),
        DimensionalData.Dim{:chain}(FlexiChains.chain_indices(chn)),
    )
end

@testset "Turing integration" begin
    schools = [
        "Choate",
        "Deerfield",
        "Phillips Andover",
        "Phillips Exeter",
        "Hotchkiss",
        "Lawrenceville",
        "St. Paul's",
        "Mt. Hermon",
    ]

    school_dim = Dim{:school}(schools)
    y = DimArray([28.0, 8.0, -3.0, 7.0, -1.0, 1.0, 18.0, 12.0], school_dim)
    σ = DimArray([15.0, 10.0, 16.0, 11.0, 9.0, 11.0, 10.0, 18.0], school_dim)

    @model function noncentered_eight(σ; dim=only(dims(σ)), n=length(σ))
        μ ~ Normal(0, 5)
        τ ~ truncated(Cauchy(0, 5); lower=0)
        θ_tilde ~ withdims(filldist(Normal(), n), dim)
        θ := @. μ + τ * θ_tilde
        y ~ withdims(arraydist(Normal.(θ, σ)), dim)
    end

    model = noncentered_eight(σ)
    model_cond = model | (; y)

    @testset "rand" begin
        params = rand(model)
        @testset for var_name in (:θ_tilde, :y)
            @test params[var_name] isa AbstractDimArray
            @test Dimensions.dimsmatch(Dimensions.dims(params[var_name]), (school_dim,))
        end
    end

    @testset "sample" begin
        ndraws = 1_000
        @testset for sampler in (NUTS(), MH(), Prior()), nchains in (1, 4)
            kws = (; chain_type=VNChain, progress=false, verbose=false)
            if nchains == 1
                chn = sample(model, sampler, ndraws; kws...)
            else
                chn = sample(model, sampler, MCMCThreads(), ndraws, nchains; kws...)
            end
            @test chn isa VNChain
            @test size(chn) == (ndraws, nchains)
            sample_dims = flexichains_sample_dims(chn)
            @testset for var_name in (:μ, :τ)
                var = chn[@varname($(var_name))]
                @test var isa AbstractDimArray
                @test Dimensions.dimsmatch(Dimensions.dims(var), sample_dims)
            end
            @testset for var_name in (:θ_tilde, :θ, :y)
                var = chn[@varname($(var_name))]
                @test var isa AbstractDimArray
                @test Dimensions.dimsmatch(
                    Dimensions.dims(var), (sample_dims..., school_dim)
                )
            end
        end

        @testset "returned" begin
            chn = sample(
                model,
                Prior(),
                MCMCThreads(),
                1000,
                4;
                chain_type=VNChain,
                progress=false,
                verbose=false,
            )
            sample_dims = flexichains_sample_dims(chn)
            y_returned = returned(model, chn)
            @test y_returned isa AbstractDimArray
            @test_broken Dimensions.dimsmatch(
                Dimensions.dims(y_returned), (sample_dims..., school_dim)
            )
        end

        @testset "predict" begin
            chn = sample(
                model_cond,
                NUTS(),
                MCMCThreads(),
                1000,
                4;
                chain_type=VNChain,
                progress=false,
                verbose=false,
            )
            sample_dims = flexichains_sample_dims(chn)
            y_pred = predict(model, chn)[@varname(y)]
            @test y_pred isa AbstractDimArray
            @test Dimensions.dimsmatch(
                Dimensions.dims(y_pred), (sample_dims..., school_dim)
            )
        end
    end
end
