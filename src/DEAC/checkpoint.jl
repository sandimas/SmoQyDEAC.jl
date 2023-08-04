

function find_checkpoint(params::DEACParameters)
    file = params.checkpoint_directory*"/DEAC_checkpoint.jld2"
    check_exists = isfile(file)
    if check_exists
        check_dict = FileIO.load(file)
        return true, check_dict
    else
        return false, nothing
    end
end

function compare_checkpoint(checkpoint_dict,params::DEACParameters,correlation_function::AbstractVector,correlation_error::AbstractVector)
    check_params = checkpoint_dict["params"]
    cor_dat = checkpoint_dict["corr_data"]
    cor_err = checkpoint_dict["corr_err"]
    return check_params == params && cor_dat == correlation_function && cor_err == correlation_error
end

function delete_checkpoint(params::DEACParameters)
    file = params.checkpoint_directory*"/DEAC_checkpoint.jld2"
    if isfile(file)
        rm(file)
    end
end


function save_checkpoint(bin_data, bin_err, bin_num, params::DEACParameters,correlation_function::AbstractArray,correlation_error::AbstractArray,zeroth_momentum::AbstractArray)
    file = params.checkpoint_directory*"/DEAC_checkpoint.jld2"
    chk_data = Dict{String,Any}(
        "bin_data" => bin_data,
        "bin_error" => bin_err,
        "bin_num" => bin_num,
        "params" => params,
        "corr_data" => correlation_function,
        "corr_err" => correlation_error,
        "zeroth" => zeroth_momentum
    )
    FileIO.save(file,chk_data)
end
