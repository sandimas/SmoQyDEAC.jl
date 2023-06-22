using CairoMakie
using Interpolations

@doc raw"""
    run_DEAC(correlation_function::AbstractVector,
            correlation_function_error::AbstractVector,
            β::Float64,
            input_grid::Vector{Float64},
            out_ωs::Vector{Float64},
            kernel_type::String,
            num_bins::Int64,
            runs_per_bin::Int64,
            output_file::String
            checkpoint_directory::String;
            population_size::Int64=8,
            base_seed::Int64=8675309,
            crossover_probability::Float64=0.9,
            self_adapting_crossover_probability::Float64=0.1,
            differential_weight::Float64=0.9,
            self_adapting_differential_weight_probability::Float64=0.1,
            self_adapting_differential_weight::Float64=0.9,
            stop_minimum_fitness::Float64=1.0,
            number_of_generations::Int64=100000)

Runs the DEAC algorithm on data passed in `correlation_function` using error by `correlation_function_error`.

`input_grid` will either be an evenly spaced set of imaginary time slices including 0 and β, or a set of matsubara frequencies.

Allowed kernel types include \"freq_fermionic_real\", \"freq_fermionic_imaginary\",
\"freq_bosonic_real\", \"freq_bosonic_imaginary\", \"time_bosonic\", \"time_fermionic\", or \"time_bosonic_symmetric\"
\"time_fermionic_antisymmetric\" is a WIP
"""
function DEAC_sc(correlation_function::AbstractVector,
                  correlation_function_error::AbstractVector,
                  β::Float64,
                  input_grid::Vector{Float64},
                  out_ωs::Vector{Float64},
                  kernel_type::String,
                  num_bins::Int64,
                  runs_per_bin::Int64,
                  output_file::String,
                  checkpoint_directory::String;

                  population_size::Int64=8,
                  base_seed::Integer=8675309,
                  crossover_probability::Float64=0.9,
                  self_adapting_crossover_probability::Float64=0.1,
                  differential_weight::Float64=0.9,
                  self_adapting_differential_weight_probability::Float64=0.1,
                  self_adapting_differential_weight::Float64=0.9,
                  stop_minimum_fitness::Float64=1.0,
                  number_of_generations::Int64=100000,
                  autoresume_from_checkpoint=true,
                  keep_bin_data=true
                )
    #
    params = DEACParameters(β,input_grid,out_ωs,kernel_type,output_file,checkpoint_directory,
                            num_bins,runs_per_bin,population_size,base_seed,
                            crossover_probability,self_adapting_crossover_probability,
                            differential_weight,self_adapting_differential_weight_probability,
                            self_adapting_differential_weight,stop_minimum_fitness,number_of_generations)
    #
    return run_DEAC_sc(correlation_function,correlation_function_error,params,autoresume_from_checkpoint,keep_bin_data)
end    

# Run the DEAC algorithm
function run_DEAC_sc(_correlation_function::AbstractVector,
                  correlation_function_error::AbstractVector,
                  params::DEACParameters,
                  autoresume_from_checkpoint::Bool,
                  keep_bin_data::Bool)
    
    # Assert parameters are within allowable/realistic ranges
    @assert params.population_size >= 6 # DEAC can be run with as few as 4, but it gives garbage results
    @assert params.β > 0.0
    @assert params.num_bins > 1
    @assert params.runs_per_bin >= 1
    @assert params.kernel_type in allowable_kernels
    @assert params.crossover_probability > 0.0 && params.crossover_probability < 1.0
    @assert params.self_adapting_crossover_probability > 0.0 && params.self_adapting_crossover_probability < 1.0
    @assert params.differential_weight > 0.0 && params.differential_weight < 1.0
    @assert params.self_adapting_differential_weight > 0.0 && params.self_adapting_differential_weight < 1.0
    @assert params.self_adapting_differential_weight_probability > 0.0 && params.self_adapting_differential_weight_probability < 1.0
    @assert params.stop_minimum_fitness > 0.0
    @assert params.number_of_generations >= 1
    @assert params.base_seed >= 1
   
    oddness_param = 0.
    flips_param = 0.1
    tension_param = 0.1
    start_bin = 1


    itp = Interpolations.interpolate(_correlation_function,BSpline(Quadratic(Reflect(OnCell()))))
    
    dc = zeros(Float64,size(_correlation_function,1))
    for i in 1:size(_correlation_function,1)
        
        if i == 1
            x = 1.01
        elseif i == 101
            x = 100.99
        else
            x = i
        end
        dc[i] = Interpolations.gradient(itp,x)[1]
    end
    # println(itp(1),itp(101))
    # dc = Interpolations.gradient(itp,2:size(_correlation_function,1)-1)
    correlation_function = correlation_function .- dc



    calculated_zeroth_moment = zeros(Float64,(1,params.num_bins))

    bin_data = zeros(Float64,(size(params.out_ωs,1),params.num_bins))
    bin_error = zeros(Float64,(size(params.out_ωs,1),params.num_bins))
    data_check = zeros(Float64,(2,params.num_bins,params.runs_per_bin,size(params.out_ωs,1)))
    # Checkpoint
    if autoresume_from_checkpoint
        chk_exists, chk_dict = find_checkpoint(params)
        if chk_exists
            if compare_checkpoint(chk_dict,params,correlation_function,correlation_function_error)
                println("Checkpoint found at "*params.checkpoint_directory*"/DEAC_checkpoint.jld2")
                start_bin = chk_dict["bin_num"] + 1
                println("Parameters match. Resuming at bin ",start_bin,"\n")
                bin_data = chk_dict["bin_data"]
                bin_error = chk_dict["bin_error"]
                calculated_zeroth_moment = chk_dict["zeroth"]
            else
                println("Checkpoint found at "*params.checkpoint_directory*"/DEAC_checkpoint.jld2")
                println("Mismatched parameters. Exiting")
                exit()
            end
        end
    end


    # Zeroth moments to check results
    zeroth_moment = 0
    zeroth_moment_err = 0.0
    Δω = (params.out_ωs[end]-params.out_ωs[1])/(size(params.out_ωs,1))
    norm_factor = Vector{Float64}(undef,size(params.out_ωs,1))
    norm_factor .= 1.0

    if occursin("time",params.kernel_type)
        if occursin("_symmetric",params.kernel_type)
            zeroth_moment = correlation_function[1] 
            zeroth_moment_err = correlation_function_error[1]
            for ω in 1:size(params.out_ωs,1)
                if params.out_ωs[ω] ≈ 0.0
                    norm_factor[ω] = 1.0/params.β
                else
                    norm_factor[ω] = 0.5*params.out_ωs[ω]*coth(0.5*params.out_ωs[ω]*params.β)
                end
            end
        elseif occursin("bosonic",params.kernel_type)
            for ω in 1:size(params.out_ωs,1)
                if params.out_ωs[ω] ≈ 0.0
                    norm_factor[ω] = 1.0/(params.β+1)
                else
                    norm_factor[ω] = params.out_ωs[ω]/(1-exp(-params.β*params.out_ωs[ω]))
                end
            end
            zeroth_moment = correlation_function[1]
            zeroth_moment_err = correlation_function_error[1]
        elseif occursin("antisymmetric",params.kernel_type)
            zeroth_moment = correlation_function[1]
            zeroth_moment_err = correlation_function_error[1]
            for ω in 1:size(params.out_ωs,1)
                norm_factor[ω] = tanh(0.5*params.out_ωs[ω]*params.β)
            end
        else
            zeroth_moment = correlation_function[1] + correlation_function[end]
            zeroth_moment_err = sqrt(correlation_function_error[1]^2 + correlation_function_error[end]^2)
        end
    else
        # Fixme
        zeroth_moment = 1.0
    end

    # Utilize the correct kernel
    K = generate_K2(params)
    generations_total = 0
    # loop over bins
    for bin in start_bin:params.num_bins
        
        generations = zeros(Int64,params.runs_per_bin)

        # Multithread each run per bin. 
        run_data = zeros(Float64,(size(params.out_ωs,1),params.runs_per_bin))
        Threads.@threads for run in 1:params.runs_per_bin

            # Each run utilizes its own RNG with a unique seed
            seed = params.base_seed + (bin - 1) * params.runs_per_bin + run
            rng = Random.Xoshiro(seed)
            
            # Randomly set initial population and normalize it
            population_old_p  = reshape(Random.rand(rng,size(params.out_ωs,1)*params.population_size),(size(params.out_ωs,1),params.population_size))
            population_old_n  = reshape(Random.rand(rng,size(params.out_ωs,1)*params.population_size),(size(params.out_ωs,1),params.population_size))
            
            population_new_p = zeros(Float64,(size(params.out_ωs,1),params.population_size))
            population_new_n = zeros(Float64,(size(params.out_ωs,1),params.population_size))
            
            # Get model Fitness
            model = *(K[:,:],population_old_p-population_old_n)
            fitness_old = Χ²(correlation_function,model,correlation_function_error) ./ size(params.input_grid,1) + 
                          oddness_param*calc_oddness(population_old_n,population_old_p) + flips_param * sign_flips(population_old_n,population_old_p) +
                          tension_param * tension(population_old_n,population_old_p)
            
            # Set initial parameters for algo
            crossover_probability_new = zeros(Float64,params.population_size)
            crossover_probability_old = zeros(Float64,params.population_size)
            crossover_probability_old .= params.crossover_probability

            differential_weights_new = zeros(Float64,params.population_size)
            differential_weights_old = zeros(Float64,params.population_size)
            differential_weights_old .= params.differential_weight
            numgen = 0

            # Loop over generations until number_of_generations or fitness is achieved
            for gen in 1:params.number_of_generations
            
                # If fitness achieved, exit loop
                minimum_fitness = minimum(fitness_old)
                if (minimum_fitness <= params.stop_minimum_fitness) 

                    break
                end
            
                # Modify DEAC parameters stochastically
                for pop in 1:params.population_size
                    crossover_probability_new[pop] = (Random.rand(rng,Float64)<params.self_adapting_crossover_probability) ? rand(rng,Float64) : crossover_probability_old[pop]
                    differential_weights_new[pop] = (Random.rand(rng,Float64)<params.self_adapting_differential_weight_probability) ? 2.0*rand(rng,Float64) : differential_weights_old[pop]
                end

                # Randomly set some ω points to 'mutate'
                mutate_indices_rnd_p = Random.rand(rng,Float64, (params.population_size,size(params.out_ωs,1))) 
                mutate_indices_rnd_n = Random.rand(rng,Float64, (params.population_size,size(params.out_ωs,1))) 
                mutate_indices_p = Array{Bool}(undef,(params.population_size,size(params.out_ωs,1)))
                mutate_indices_n = Array{Bool}(undef,(params.population_size,size(params.out_ωs,1)))
                for pop in 1:params.population_size
                    mutate_indices_p[pop,:] = mutate_indices_rnd_p[pop,:] .< crossover_probability_new[pop]
                    mutate_indices_n[pop,:] = mutate_indices_rnd_n[pop,:] .< crossover_probability_new[pop]
                end
            
                # Set triplet of other populations for mutations
                mutant_indices_p = get_mutant_indices(rng,params.population_size)
                mutant_indices_n = get_mutant_indices(rng,params.population_size)
                
                # if mutate_indices, do mutation, else keep same
                for pop in 1:params.population_size
                    for ω in 1:size(params.out_ωs,1)
                        ω1 = ω
                        ω2 = ω
                        if ω != 1
                            ω1 = ω-1
                        end
                        if ω != size(params.out_ωs,1)
                            ω2 = ω+1
                        end

                        if mutate_indices_p[pop,ω]
                            population_new_p[ω,pop] = abs(population_old_p[ω,mutant_indices_p[1,pop]] + differential_weights_new[pop]*
                                                        (population_old_p[ω1,mutant_indices_p[2,pop]]-population_old_p[ω2,mutant_indices_p[3,pop]]))
                        else
                            population_new_p[ω,pop] = population_old_p[ω,pop]
                        end
                        if mutate_indices_n[pop,ω]
                            population_new_n[ω,pop] = abs(population_old_n[ω,mutant_indices_n[1,pop]] + differential_weights_new[pop]*
                                                        (population_old_n[ω1,mutant_indices_n[2,pop]]-population_old_n[ω2,mutant_indices_n[3,pop]]))
                        else
                            population_new_n[ω,pop] = population_old_n[ω,pop]
                        end
                    end
                end
                
                model = *(K[:,:],population_new_p-population_new_n)
            
                fitness_new = Χ²(correlation_function,model,correlation_function_error) ./ size(params.input_grid,1)  + 
                              oddness_param*calc_oddness(population_new_n,population_new_p) + flips_param * sign_flips(population_new_n,population_new_p) +
                              tension_param*tension(population_old_n,population_old_p)
                
                for pop in 1:params.population_size
                    if fitness_new[pop] <= fitness_old[pop]
                        fitness_old[pop] = fitness_new[pop]
                        crossover_probability_old[pop] = crossover_probability_new[pop]
                        differential_weights_old[pop] = differential_weights_new[pop]
                        population_old_p[:,pop] = population_new_p[:,pop]
                        population_old_n[:,pop] = population_new_n[:,pop]
                    end
                end
                # println("fitness: ",fitness_old[1])
                numgen = numgen + 1
                
            end # generations
            fit, fit_idx = findmin(fitness_old)
            generations[run] = numgen
            if numgen == params.number_of_generations
                println("fit,", fit, "\t numgen,", numgen)
            end    
            # Testing info
            
            run_data[:,run] = population_old_p[:,fit_idx] - population_old_n[:,fit_idx]
            data_check[1,bin,run,:] = population_old_p[:,fit_idx]
            data_check[2,bin,run,:] = population_old_n[:,fit_idx]

           
            try
                mkdir("plotss")
            catch
            end
            f = Figure()
            ax=Axis(f[1,1])
            lines!(params.out_ωs,run_data[:,run])
            f
            save("plotss/"*string(bin)*"_"*string(run)*".png",f)

            println("stats")
            println(" signflips: ",sign_flips(population_old_n,population_old_p) )
            println(" tension:   ",tension(population_old_n,population_old_p))
            # println(" oddness:   ",calc_oddness(population_old_n,population_old_p)) 
        end # run per bin
        bin_error[:,bin] = Statistics.std(run_data,dims=2)
        bin_data[:,bin] = Statistics.mean(run_data,dims=2)
        generations_total += sum(generations)
        
        # Bosonic time kernels steal a factor of ω from the spectral function.
        # Multiply it back in if needed
        
        calculated_zeroth_moment[1,bin] = sum(bin_data[:,bin] .* Δω .* norm_factor)

        if  occursin("bosonic",params.kernel_type)
            bin_data[:,bin] = bin_data[:,bin] .* params.out_ωs
        end
        
        println("Finished bin ",bin," of ",params.num_bins)
       
        if bin != params.num_bins
            save_checkpoint(bin_data,bin_error,bin,params,correlation_function,correlation_function_error,calculated_zeroth_moment)
        end
        
    end # bins
    data_check_out = Dict{String,Any}( "data" => data_check)
    FileIO.save("sc_bin_dat.jld2",data_check_out)
    zero_avg, zero_err = jackknife(calculated_zeroth_moment)
    gen_per_run = generations_total/(params.num_bins * params.runs_per_bin)
    differential = 100.0*abs(zeroth_moment-zero_avg[1])/zeroth_moment
    
    # Merge data, save it, pass it back to user
    println("\nSaving data to ",params.output_file," and deleting checkpoint file\n")
    
    println("Run Statistics")
    println(@sprintf(" Expected 0th moment:   %01.3f ± %01.3f",zeroth_moment,zeroth_moment_err) )
    println(@sprintf(" DEAC 0th moment:       %01.3f ± %01.3f",zero_avg[1],zero_err[1]))
    println(@sprintf(" 0th moment difference: %01.3f%%",differential))
    println(@sprintf(" Mean generations/run:  %01.3f",gen_per_run))
    println(" ")
    data, err = jackknife(bin_data)
    if keep_bin_data
        bin_dict = Dict{String,Any}(
            "A" => data,
            "σ" => err,
            "ωs" => params.out_ωs,
            "zeroth" => zero_avg[1],
            "expected_zeroth" => zeroth_moment,
            "zerothσ" => zero_err[1],
            "avg_generations" => gen_per_run,
            "bin" => bin_data,
            "σbin" => bin_error,
            "0thbin" => calculated_zeroth_moment
        )
    else
        bin_dict = Dict{String,Any}(
            "A" => data,
            "σ" => err,
            "zeroth" => zero_avg[1],
            "expected_zeroth" => zeroth_moment,
            "zerothσ" => zero_err[1],
            "avg_generations" => gen_per_run,
            "ωs" => params.out_ωs
        )
    end
    FileIO.save(params.output_file,bin_dict)
    delete_checkpoint(params)
    return bin_dict
end # run_DEAC()



function generate_K2(params::DEACParameters)
    nω = size(params.out_ωs,1)
    ngrid = size(params.input_grid,1)
    K = zeros(Float64,(ngrid,nω))
    dgrid = params.input_grid[2]-params.input_grid[1]

    
    for ω in 1:nω
        nf = 1 + exp(-params.out_ωs[ω]*params.β)    
        for τ in 1:ngrid
            
            K[τ,ω] = dgrid*(exp(-params.out_ωs[ω]*params.input_grid[τ])*(1+params.out_ωs[ω])-(1-1+params.out_ωs[ω])*exp(-(params.β-params.input_grid[τ])*params.out_ωs[ω])) / nf
            
            
        end
    end
    
    return K
end

function calc_oddness(pops_p,pops_n)
    nω=size(pops_p,1)

    npop = size(pops_p,2)
    diff = zeros(Float64,npop)
    for pop in 1:npop
        for ω in 1:nω
            diff[pop] += abs(pops_p[ω,pop]-pops_n[1+nω-ω,pop])
        end
    end
    return diff ./ nω
end

function sign_flips(pops_p,pops_n)
    nω=size(pops_p,1)

    npop = size(pops_p,2)
    flips = zeros(Float64,npop)
    sign_old = (pops_p[1,:]-pops_n[1,:]) .>= 0.0
    for pop in 1:npop
        for ω in 2:nω
            sign_new = (pops_p[ω,pop]-pops_n[ω,pop]) >= 0.0
            if (sign_old[pop] ⊻ sign_new)
                flips[pop] += 1.0
            end
            sign_old[pop] = sign_new
        end
    end
    return flips
end

function tension(pops_p,pops_n)
    nω=size(pops_p,1)
    npop = size(pops_p,2)
    tension = zeros(Float64,npop)
    for pop in 1:npop
        for ω in 2:nω
            tension[pop] += abs(pops_p[ω,pop]-pops_p[ω-1,pop]) +abs(pops_n[ω,pop]-pops_n[ω-1,pop])
        end
    end
    return tension ./ nω
end
