using Statistics

function do_correct_drift(dat, num_directions; verbose=false)
    res = copy(dat) # make a copy to avoid modifying the original data
    slices_per_direction = size(dat, ndims(dat)) ÷ num_directions
    wf_proj = sum(slice(dat,ndims(dat),1:slices_per_direction), dims=ndims(dat))
    # wf_proj = FindShift.damp_edge_outside(wf_proj)
    for d in 2:num_directions
        myslice = slice(dat, ndims(dat),1+(d-1)*slices_per_direction:d*slices_per_direction)
        res_slice = slice(res, ndims(dat),1+(d-1)*slices_per_direction:d*slices_per_direction)
        wf_proj_ts = sum(myslice, dims=ndims(dat))
        # wf_proj_ts = FindShift.damp_edge_outside(wf_proj_ts)
        shift_vec = find_shift_iter(wf_proj, wf_proj_ts)
        if (verbose)
            println("direction $d has shift $(shift_vec)")
        end
        res_slice .= FindShift.shift(myslice, shift_vec)
    end
    return res
end


"""
    preprocess_sim(dat; bg = 100f0, num_directions=nothing)

Preprocesses the data for SIM analysis by subtracting a background value and normalizing the data.
"""
function preprocess_sim(dat; bg = 100f0, num_directions=nothing, reg_const=1e-6, correct_drift=true, enforce_mean=true, verbose=false)
    dat = dat .- bg # copies
    # correct slice brightness fluctuations
    dat .*= mean(dat) ./ mean(dat, dims=(1:ndims(dat)-1))

    if !isnothing(num_directions)
        if (correct_drift)
            dat = do_correct_drift(dat, num_directions; verbose=verbose)
        end
        if (enforce_mean)
            all_mean =  mean(dat, dims=ndims(dat))
            num_phases = size(dat, ndims(dat)) ÷ num_directions
            for d in 1:num_directions
                sub_data = slice(dat, ndims(dat), (d-1)*num_phases+1:d*num_phases)
                sub_mean = mean(sub_data, dims=ndims(sub_data))
                sub_data .*= all_mean .* sub_mean ./ (abs2.(sub_mean) .+ reg_const)
            end
        end
    end
    return dat
end
