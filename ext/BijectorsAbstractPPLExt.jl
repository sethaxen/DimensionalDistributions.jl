module BijectorsAbstractPPLExt

using AbstractPPL: AbstractPPL
using Bijectors: Bijectors
using DimensionalDistributions: DimensionalDistributions

@static if isdefined(Bijectors, :VectorBijectors) &&
    isdefined(AbstractPPL, Symbol("@opticof"))
    using Bijectors: VectorBijectors

    function VectorBijectors.optic_vec(
        d::DimensionalDistributions.AbstractDimArrayDistribution
    )
        return [AbstractPPL.@opticof(_[i]) for i in Iterators.product(axes(d)...)]
    end

    function VectorBijectors.linked_optic_vec(
        d::DimensionalDistributions.AbstractDimArrayDistribution
    )
        return VectorBijectors.optic_vec(d)
    end

    function VectorBijectors.linked_optic_vec(
        d::DimensionalDistributions.AsDimArrayDistribution
    )
        return VectorBijectors.linked_optic_vec(parent(d))
    end
end

end  # module
