module SmoQyDEAC

using Random
using Statistics
using FileIO
using Printf
using LinearAlgebra

include("DEAC/types.jl")
include("DEAC/deac.jl")
include("DEAC/jackknife.jl")
include("DEAC/kernels.jl")
include("DEAC/utility.jl")
include("DEAC/checkpoint.jl")

export DEAC_Std
export DEAC_Binned

end # module SmoQyDEAC
