using DimensionalDistributions
using Test
using Aqua
using JET

@testset "DimensionalDistributions.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(DimensionalDistributions)
    end
    # JET does not support prerelease versions of Julia
    if isempty(VERSION.prerelease)
        @testset "Code linting (JET.jl)" begin
            JET.test_package(
                DimensionalDistributions; target_modules=(DimensionalDistributions,)
            )
        end
    end

    include("abstractdimarraydist.jl")
    include("asdimarraydist.jl")
end
