using DimensionalDistributions
using Documenter

DocMeta.setdocmeta!(DimensionalDistributions, :DocTestSetup, :(using DimensionalDistributions); recursive=true)

makedocs(;
    modules=[DimensionalDistributions],
    authors="Seth Axen <seth@sethaxen.com> and contributors",
    sitename="DimensionalDistributions.jl",
    format=Documenter.HTML(;
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)
