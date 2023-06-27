
# Generate Kernel
# Notably, for bosonic kernels we multiply in a factor of ω.
# This makes the kernel and the spectral function positive and analytic for all ω
# Before we return data, though, we will multiply bosonic functions by ω
function generate_K(params::DEACParameters)
    nω = size(params.out_ωs,1)
    ngrid = size(params.input_grid,1)
    K = zeros(Float64,(ngrid,nω))
    dgrid = params.input_grid[2]-params.input_grid[1]

    if params.kernel_type == "freq_bosonic"
        for ω in 1:nω
            ω2 = params.out_ωs[ω]^2
            for ωm in 1:ngrid
                # L'Hopital 0/0
                if (params.input_grid[ωm]==0.0 && params.out_ωs[ω] == 0.0)
                    K[ωm,ω] = 2.0*dgrid
                else
                    K[ωm,ω] = 2.0*dgrid*ω2 /(params.input_grid[ωm]^2+ω2)
                end
            end
        end
    elseif params.kernel_type == "freq_fermionic"
        for ω in 1:nω
            ω2 = params.out_ωs[ω]^2
            for ωm in 1:ngrid
                
                K[ωm,ω] = dgrid*params.out_ωs[ω] /(params.input_grid[ωm]^2+ω2)
            end
        end
    elseif params.kernel_type == "time_bosonic"
        nb =  n_b(params)
        for ω in 1:nω
            for τ in 1:ngrid
                K[τ,ω] = dgrid*exp(-params.out_ωs[ω]*params.input_grid[τ]) * nb[ω]
            end
        end
    elseif params.kernel_type == "time_bosonic_symmetric"
        nb = n_b(params)
        for ω in 1:nω
            for τ in 1:ngrid
                K[τ,ω] = 0.5*dgrid*(exp(-params.out_ωs[ω]*params.input_grid[τ]) + exp(-params.out_ωs[ω]*(params.β - params.input_grid[τ]))) * nb[ω]
            end
        end
    elseif params.kernel_type == "time_fermionic"
        for ω in 1:nω
            for τ in 1:ngrid
                K[τ,ω] = dgrid / (exp(params.out_ωs[ω] * params.input_grid[τ]) + exp(-params.out_ωs[ω] * (params.β - params.input_grid[τ])))
            end
        end
        
    elseif params.kernel_type == "time_fermionic_antisymmetric"
        println("time_fermionic_antisymmetric NOT YET IMPLEMENTED")
        exit()
        for ω in 1:nω
            for τ in 1:ngrid
                K[τ,ω] = 0.5* dgrid * (exp(-params.out_ωs[ω] * params.input_grid[τ]) - exp(-params.out_ωs[ω] *(params.β - params.input_grid[τ]))) / (1 + exp(-params.β * params.out_ωs[ω]))
            end
        end
    end
    return K
end


# Bose factor * ω, 
function n_b(params::DEACParameters)
    close = 1.0e-6
    nω = size(params.out_ωs,1)
    arr = zeros(Float64,nω)
    for ω in 1:nω
        # L'hopital
        if abs(params.out_ωs[ω]) < close
            arr[ω] = 1.0 /params.β
        else
            arr[ω] = params.out_ωs[ω] / (1 - exp(-params.β * params.out_ωs[ω]))
        end
    end 
    return arr
end
