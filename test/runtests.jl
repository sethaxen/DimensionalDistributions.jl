using Aqua
using DimensionalDistributions
using JET
using Test
using Turing: Turing  # trigger extension loading

@testset "DimensionalDistributions.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(DimensionalDistributions)
        @testset for pkg in [:Bijectors, :DynamicPPL]
            ext = Base.get_extension(DimensionalDistributions, Symbol(pkg, "Ext"))
            Aqua.test_all(
                ext;
                piracies=(;
                    treat_as_own=[
                        DimensionalDistributions.AbstractDimArrayDistribution,
                        DimensionalDistributions.AsDimArrayDistribution,
                    ]
                ),
            )
        end
    end
    @testset "Code linting (JET.jl)" begin
        JET.test_package(
            DimensionalDistributions; target_modules=(DimensionalDistributions,)
        )
    end

    include("abstractdimarraydist.jl")
    include("asdimarraydist.jl")
end
