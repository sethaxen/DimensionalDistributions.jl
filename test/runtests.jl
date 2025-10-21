using DimensionalDistributions
using Test
using Aqua
using JET

@testset "DimensionalDistributions.jl" begin
    @testset "Code quality (Aqua.jl)" begin
        Aqua.test_all(DimensionalDistributions)
    end
    @testset "Code linting (JET.jl)" begin
        JET.test_package(DimensionalDistributions; target_defined_modules = true)
    end
    # Write your tests here.
end
