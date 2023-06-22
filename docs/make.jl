using SmoQyDEAC
using Documenter

DocMeta.setdocmeta!(SmoQyDEAC, :DocTestSetup, :(using SmoQyDEAC); recursive=true)

makedocs(;
    modules=[SmoQyDEAC],
    authors="James Neuhaus <james.neuhaus@gmail.com>",
    repo="https://github.com/sandimas/SmoQyDEAC.jl/blob/{commit}{path}#{line}",
    sitename="SmoQyDEAC.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://sandimas.github.io/SmoQyDEAC.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/sandimas/SmoQyDEAC.jl",
    devbranch="main",
)
