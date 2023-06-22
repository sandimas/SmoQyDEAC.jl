module SmoQyDEAC

using Random
using Statistics
using FileIO
using Printf
# using Documenter
include("DEAC/types.jl")
include("DEAC/deac.jl")
include("DEAC/jackknife.jl")
include("DEAC/kernels.jl")
include("DEAC/utility.jl")
include("DEAC/checkpoint.jl")
# include("DEAC/deac_sc.jl")

export DEAC

# export DEAC_sc
end # module SmoQyDEAC
