@doc raw"""
    DEAC_Std(correlation_function::AbstractVector,
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
         stop_minimum_fitness::Float64=1.0,
         number_of_generations::Int64=100000,
         keep_bin_data=true,
         autoresume_from_checkpoint=true,
         
         crossover_probability::Float64=0.9,
         self_adapting_crossover_probability::Float64=0.1,
         differential_weight::Float64=0.9,
         self_adapting_differential_weight_probability::Float64=0.1,
         self_adapting_differential_weight::Float64=0.9)
    
Runs the DEAC algorithm on data passed in `correlation_function` using $\Chi^2$ fitting from the error passed in by `correlation_function_error`.
# Arguments
- `correlation_function::AbstractVector`: Input data in τ space 
- `correlation_function_error::AbstractVector`: Error associated with input data
- `β::Float64`: Inverse temperature
- `input_grid::Vector{Float64}`: Evenly spaced values in τ from 0 to β, including end points
- `out_ωs::Vector{Float64}`: Energies for AC output
- `kernel_type::String`: See below for allowable kernels
- `num_bins::Int64`: Bins to generate
- `runs_per_bin::Int64`: Number of runs per bin for statistics
- `output_file::String`: File to store output dictionary. jld2 format recommended
- `checkpoint_directory::String`: Directory to store checkpoint data. 

# Optional Arguments
- `population_size::Int64=8`: DEAC population size. Must be ≥ 6
- `base_seed::Int64=8675309`: Seed
- `stop_minimum_fitness::Float64=1.0`: Value below which fit is considered good
- `number_of_generations::Int64=100000`: Maximum number of mutation loops
- `keep_bin_data::Bool=true`: Save binned data or not
- `autoresume_from_checkpoint::Bool=true`: Resume from checkpoint if possible

# Optional algorithm arguments
- `crossover_probability::Float64=0.9`: Starting likelihood of crossover
- `self_adapting_crossover_probability::Float64=0.1`: Chance of crossover probability changing
- `differential_weight::Float64=0.9`: Weight for second and third mutable indices
- `self_adapting_differential_weight_probability::Float64=0.1`: Likelihood of SAD changing
- `self_adapting_differential_weight::Float64=0.9`: SAD

Each run will use its own seed. E.g. if you run 10 bins with 100 runs per bin, you will use seeds `base_seed:base_seed+999`. 
You may increment your base seed by 1000, use another output file name, and generate more statistics later.

# Output
SmoQyDEAC will save a dictionary to the file name passed via the `output_file` argument. The same dictionary will be returned by the function.

# Checkpointing
SmoQyDEAC will place a file named `DEAC_checkpoint.jld2` in the directory passed in `checkpoint_directory`. After completing every bin this file will be saved.
After the last bin is finished the file will be deleted. When `autoresume_from_checkpoint=true` SmoQyDEAC will attempt to resume from the checkpoint. 
If the arguments passed do not match those in the checkpoint the code will exit.
"""
function DEAC_Std(correlation_function::AbstractVector,
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
                  autoresume_from_checkpoint=false,
                  keep_bin_data=true,
                  W_ratio_max = 1.0e6
                )
    #
    println("\n*** It is highly recommended to use binned data and the covariant matrix method instead (DEAC_Binned) if possible ***\n")
    params = DEACParameters(β,input_grid,out_ωs,kernel_type,output_file,checkpoint_directory,
                            num_bins,runs_per_bin,population_size,base_seed,
                            crossover_probability,self_adapting_crossover_probability,
                            differential_weight,self_adapting_differential_weight_probability,
                            self_adapting_differential_weight,stop_minimum_fitness,number_of_generations)
    #
    return run_DEAC((correlation_function,correlation_function_error),params,autoresume_from_checkpoint,keep_bin_data,W_ratio_max)
end    


@doc raw"""
    DEAC_Binned(correlation_function::AbstractMatrix,
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
         stop_minimum_fitness::Float64=1.0,
         number_of_generations::Int64=100000,
         keep_bin_data=true,
         autoresume_from_checkpoint=false,
         
         crossover_probability::Float64=0.9,
         self_adapting_crossover_probability::Float64=0.1,
         differential_weight::Float64=0.9,
         self_adapting_differential_weight_probability::Float64=0.1,
         self_adapting_differential_weight::Float64=0.9,
         )
    
Runs the DEAC algorithm on data passed in `correlation_function` using $\Chi^2$ fitting from the error passed in by `correlation_function_error`.
# Arguments
- `correlation_function::AbstractVector`: Input data in τ space 
- `correlation_function_error::AbstractVector`: Error associated with input data
- `β::Float64`: Inverse temperature
- `input_grid::Vector{Float64}`: Evenly spaced values in τ from 0 to β, including end points
- `out_ωs::Vector{Float64}`: Energies for AC output
- `kernel_type::String`: See below for allowable kernels
- `num_bins::Int64`: Bins to generate
- `runs_per_bin::Int64`: Number of runs per bin for statistics
- `output_file::String`: File to store output dictionary. jld2 format recommended
- `checkpoint_directory::String`: Directory to store checkpoint data. 

# Optional Arguments
- `population_size::Int64=8`: DEAC population size. Must be ≥ 6
- `base_seed::Int64=8675309`: Seed
- `stop_minimum_fitness::Float64=1.0`: Value below which fit is considered good
- `number_of_generations::Int64=100000`: Maximum number of mutation loops
- `keep_bin_data::Bool=true`: Save binned data or not
- `autoresume_from_checkpoint::Bool=true`: Resume from checkpoint if possible

# Optional algorithm arguments
- `crossover_probability::Float64=0.9`: Starting likelihood of crossover
- `self_adapting_crossover_probability::Float64=0.1`: Chance of crossover probability changing
- `differential_weight::Float64=0.9`: Weight for second and third mutable indices
- `self_adapting_differential_weight_probability::Float64=0.1`: Likelihood of SAD changing
- `self_adapting_differential_weight::Float64=0.9`: SAD
- `W_ratio_max::Float64=1.0e6`: Χ² ~ 1.0/σ², this parameter prevents [near] singularities for very small σ 
- `bootstrap_bins::Int=0`: The algorithm requires more bins than τ steps. We use bootstrapping to get 5 * nτ bins by default. User may set this higher 
                  

Each run will use its own seed. E.g. if you run 10 bins with 100 runs per bin, you will use seeds `base_seed:base_seed+999`. 
You may increment your base seed by 1000, use another output file name, and generate more statistics later.

# Output
SmoQyDEAC will save a dictionary to the file name passed via the `output_file` argument. The same dictionary will be returned by the function.

# Checkpointing
SmoQyDEAC will place a file named `DEAC_checkpoint.jld2` in the directory passed in `checkpoint_directory`. After completing every bin this file will be saved.
After the last bin is finished the file will be deleted. When `autoresume_from_checkpoint=true` SmoQyDEAC will attempt to resume from the checkpoint. 
If the arguments passed do not match those in the checkpoint the code will exit.
"""
function DEAC_Binned(correlation_function::AbstractMatrix,
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
                  autoresume_from_checkpoint=false,
                  keep_bin_data=true,
                  W_ratio_max = 1.0e6,
                  bootstrap_bins = 0
                )
    #

    if bootstrap_bins ≤ 0 || (size(correlation_function,1) < 5 * size(correlation_function,2))
        bootstrap_bins = max(bootstrap_bins,5*size(correlation_function,2))
        correlation_function = bootstrap_samples(correlation_function,bootstrap_bins,base_seed )
    end


    params = DEACParameters(β,input_grid,out_ωs,kernel_type,output_file,checkpoint_directory,
                            num_bins,runs_per_bin,population_size,base_seed,
                            crossover_probability,self_adapting_crossover_probability,
                            differential_weight,self_adapting_differential_weight_probability,
                            self_adapting_differential_weight,stop_minimum_fitness,number_of_generations)
    #
    return run_DEAC((correlation_function,nothing),params,autoresume_from_checkpoint,keep_bin_data,W_ratio_max)
end    

# Run the DEAC algorithm
function run_DEAC(Greens_tuple,
                  params::DEACParameters,
                  autoresume_from_checkpoint::Bool,
                  keep_bin_data::Bool,
                  W_ratio_max::Float64)
    
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

    use_binned = Greens_tuple[2] == nothing
    correlation_function = Greens_tuple[1]

    start_bin = 1

    bin_data = zeros(Float64,(size(params.out_ωs,1),params.num_bins))
    bin_error = zeros(Float64,(size(params.out_ωs,1),params.num_bins))



    # Checkpoint
    if autoresume_from_checkpoint
        chk_exists, chk_dict = find_checkpoint(params)
        if chk_exists
            if compare_checkpoint(chk_dict,params,Greens_tuple)
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


    # Utilize the correct kernel
    K = generate_K(params)
    generations_total = 0

    Δω = (params.out_ωs[end]-params.out_ωs[1])/(size(params.out_ωs,1)-1)
    

    if use_binned
        ###################
        # Covariance Methods
        ###################

        Nbins_in = size(correlation_function,1)
        
        # SVD on correlation bins
        corr_avg = Statistics.mean(correlation_function,dims=1)
        svd_corr = svd(correlation_function .- corr_avg)
        sigma_corr = svd_corr.S

        # Unitary transformation matrix
        U_c = svd_corr.Vt
        
        # Inverse fit array for χ^2
        U_c1 = size(U_c,1)
        W = (2.0 * U_c1) ./ (sigma_corr .* sigma_corr) 
        
        # Deal with nearly singular matrix
        W_cap = W_ratio_max * minimum(W)
        clamp!(W,0.0,W_cap)
        
        # rotate K and corr_avg
        Kp = U_c*K
        
        corr_avg_p = zeros(Float64,U_c1)
        for i in 1:U_c1
            corr_avg_p[i] = dot(view(U_c,i,:),corr_avg)
        end

    else
        W = 1.0 ./ (Greens_tuple[2] .* Greens_tuple[2])
        W_cap = W_ratio_max * minimum(W)
        clamp!(W,0.0,W_cap)
        Kp = K
        corr_avg_p = Greens_tuple[1]

    end
    #######################################
    calculated_zeroth_moment = zeros(Float64,(1,params.num_bins))

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
            population_old  = reshape(Random.rand(rng,size(params.out_ωs,1)*params.population_size),(size(params.out_ωs,1),params.population_size))
            for pop in 1:params.population_size
                population_old[:,pop] = population_old[:,pop] ./ sum(population_old[:,pop])
            end
            # normalize population_old here
            
            population_new = zeros(Float64,(size(params.out_ωs,1),params.population_size))
            
            # Get model Fitness
            model = *(Kp,population_old)

            fitness_old = Χ²(corr_avg_p,model,W) ./ size(params.input_grid,1)
            
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
                # else 
                #     println(minimum_fitness)
                end
            
                # Modify DEAC parameters stochastically
                for pop in 1:params.population_size
                    crossover_probability_new[pop] = (Random.rand(rng,Float64)<params.self_adapting_crossover_probability) ? rand(rng,Float64) : crossover_probability_old[pop]
                    differential_weights_new[pop] = (Random.rand(rng,Float64)<params.self_adapting_differential_weight_probability) ? 2.0*rand(rng,Float64) : differential_weights_old[pop]
                end

                # Randomly set some ω points to 'mutate'
                mutate_indices_rnd = Random.rand(rng,Float64, (params.population_size,size(params.out_ωs,1))) 
                mutate_indices = Array{Bool}(undef,(params.population_size,size(params.out_ωs,1)))
                for pop in 1:params.population_size
                    mutate_indices[pop,:] = mutate_indices_rnd[pop,:] .< crossover_probability_new[pop]
                end
            
                # Set triplet of other populations for mutations
                mutant_indices = get_mutant_indices(rng,params.population_size)
                
                # if mutate_indices, do mutation, else keep same
                for pop in 1:params.population_size
                    for ω in 1:size(params.out_ωs,1)
                        if mutate_indices[pop,ω]
                            population_new[ω,pop] = abs(population_old[ω,mutant_indices[1,pop]] + differential_weights_new[pop]*
                                                        (population_old[ω,mutant_indices[2,pop]]-population_old[ω,mutant_indices[3,pop]]))
                        else
                            population_new[ω,pop] = population_old[ω,pop]
                        end
                    end
                end
                
                model = *(Kp,population_new)
                
                fitness_new = Χ²(corr_avg_p,model,W) ./ size(params.input_grid,1)
                for pop in 1:params.population_size
                    if fitness_new[pop] <= fitness_old[pop]
                        fitness_old[pop] = fitness_new[pop]
                        crossover_probability_old[pop] = crossover_probability_new[pop]
                        differential_weights_old[pop] = differential_weights_new[pop]
                        population_old[:,pop] = population_new[:,pop]
                    end
                end
                
                numgen = numgen + 1
            end # generations
            fit, fit_idx = findmin(fitness_old)
            generations[run] = numgen
            # Testing info
            # println("fit, ", fit, "\tnumgen, ", numgen)
            
            run_data[:,run] = population_old[:,fit_idx]
            
        end # run per bin
        bin_error[:,bin] = Statistics.std(run_data,dims=2)
        bin_data[:,bin] = Statistics.mean(run_data,dims=2)
        generations_total += sum(generations)
        
        calculated_zeroth_moment[1,bin] = sum(bin_data[:,bin]) .* Δω
        
        # Bosonic time kernels steal a factor of ω from the spectral function.
        # Multiply it back in if needed
        if  occursin("bosonic",params.kernel_type)
            bin_data[:,bin] = bin_data[:,bin] .* params.out_ωs
            
        end
        
        println("Finished bin ",bin," of ",params.num_bins)
        
        if bin != params.num_bins
            save_checkpoint(bin_data,bin_error,bin,params,Greens_tuple,calculated_zeroth_moment)
        end
        
    end # bins
    
    zero_avg, zero_err = jackknife(calculated_zeroth_moment)
    gen_per_run = generations_total/(params.num_bins * params.runs_per_bin)
    differential = 100.0*abs(1.0-zero_avg[1])
    
    # Merge data, save it, pass it back to user
    println("\nSaving data to ",params.output_file," and deleting checkpoint file\n")
    
    println("Run Statistics")
    if occursin("fermionic",params.kernel_type)
        println(@sprintf(" Expected 0th moment:   1.00") )
        println(@sprintf(" DEAC 0th moment:       %01.3f ± %01.3f",zero_avg[1],zero_err[1]))
        println(@sprintf(" 0th moment difference: %01.3f%%",differential))
    end
    println(@sprintf(" Mean generations/run:  %01.3f",gen_per_run))
    println(" ")
    data, err = jackknife(bin_data)
    if keep_bin_data
        bin_dict = Dict{String,Any}(
            "A" => data,
            "σ" => err,
            "ωs" => params.out_ωs,
            "zeroth_moment" => zero_avg[1],
            "zeroth_moment_σ" => zero_err[1],
            "avg_generations" => gen_per_run,
            "bin_data" => bin_data,
            "bin_σ" => bin_error,
            "bin_zeroth_moment" => calculated_zeroth_moment
        )
    else
        bin_dict = Dict{String,Any}(
            "A" => data,
            "σ" => err,
            "zeroth_moment" => zero_avg[1],
            "zeroth_moment_σ" => zero_err[1],
            "avg_generations" => gen_per_run,
            "ωs" => params.out_ωs
        )
    end
    FileIO.save(params.output_file,bin_dict)
    delete_checkpoint(params)
    return bin_dict
end # run_DEAC()



