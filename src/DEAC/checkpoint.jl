

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

function compare_checkpoint(checkpoint_dict,params::DEACParameters,G_tuple)
    check_params = checkpoint_dict["params"]
    cor_dat = checkpoint_dict["G_tuple"]
    return check_params == params && cor_dat == G_tuple
end

function delete_checkpoint(params::DEACParameters)
    file = params.checkpoint_directory*"/DEAC_checkpoint.jld2"
    if isfile(file)
        rm(file)
    end
end


function save_checkpoint(bin_data, bin_err, bin_num, params::DEACParameters,G_tuple,zeroth_momentum::AbstractArray)
    file = params.checkpoint_directory*"/DEAC_checkpoint.jld2"
    chk_data = Dict{String,Any}(
        "bin_data" => bin_data,
        "bin_error" => bin_err,
        "bin_num" => bin_num,
        "params" => params,
        "G_tuple" => G_tuple,
        "zeroth" => zeroth_momentum
    )
    FileIO.save(file,chk_data)
end
