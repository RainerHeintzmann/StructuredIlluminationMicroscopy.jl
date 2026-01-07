using Statistics

"""
    do_correct_drift(dat, num_directions; verbose=false)

Corrects drift in the data by aligning slices based on the first slice of each direction.
This function assumes that the data `dat` is organized such that each direction's slices are contiguous in the last dimension.

+ Parameters:
`dat` is expected to be a 3D array where the last dimension represents different directions.
`num_directions` specifies how many directions are present in the data.
"""
function do_correct_drift(dat, num_directions; verbose=false)
    res = copy(dat) # make a copy to avoid modifying the original data
    slices_per_direction = size(dat, ndims(dat)) ÷ num_directions
    wf_proj = sum(slice(dat,ndims(dat),1:slices_per_direction), dims=ndims(dat))
    # wf_proj = FindShift.damp_edge_outside(wf_proj)
    for d in 2:num_directions
        s_start = 1 + (d-1) * slices_per_direction
        s_end = d * slices_per_direction
        myslice = slice(dat, ndims(dat),s_start:s_end)
        res_slice = slice(res, ndims(dat),s_start:s_end)
        wf_proj_ts = sum(myslice, dims=ndims(dat))
        # wf_proj_ts = FindShift.damp_edge_outside(wf_proj_ts)
        # shift_vec = find_shift_iter(wf_proj, wf_proj_ts)
        _, shift_vec = align_stack(wf_proj_ts; ref = wf_proj, method=:FindZoomFT, damp=0.05, max_freq=1.0) # , shifts=nothing
        shift_vec = shift_vec[1];
        # shift_vec = find_shift(wf_proj, wf_proj_ts)
        if (verbose)
            println("direction $d (range $(s_start:s_end)) has shift $(shift_vec)")
        end
        res_slice .= FindShift.shift(myslice, shift_vec)
    end
    return res
end

"""
    remove_oof_light(dat; psf3d)

a routine according to the work of Wengfeng Tian et al. (Liyangyi Chen group), which low-pass filters the raw data, which should be a series of 2D sections.
The filter is calculated from the 3d psf by accounting for the in- and out-of-focus (oof) part and subtracting them.
"""
function remove_oof_light(dat; psf3d, if_slices=7, myeps = 0.001)
    midz = size(psf3d, 3) ÷ 2 + 1
    zstart = midz - if_slices ÷ 2 + 1
    zend = zstart + if_slices
    psf_if = sum(psf3d[:,:,zstart:zend], dims=3)[:,:, 1];
    psf_oof = sum(psf3d[:,:,1:zstart-1], dims=3)[:,:, 1] +  sum(psf3d[:,:,zend+1: end], dims=3)[:,:, 1];
    otf_if = rfft(ifftshift(psf_if))
    otf_oof = rfft(ifftshift(psf_oof))
    otf_sum = otf_if .+ otf_oof
    nfactor = maximum(abs.(otf_sum))

    otf_filter = 1 .- otf_oof .* conj.(otf_sum) ./ (abs2.(otf_sum) .+ nfactor^2*myeps)

    res = similar(dat)
    for (res_slice, slice) in zip(eachslice(res, dims=ndims(res)), eachslice(dat, dims=ndims(res)))
        res_slice .= irfft(rfft(slice) .* otf_filter, size(res_slice,1))
    end

    return res # , otf_filter, otf_if
end

"""
    preprocess_sim(dat; bg = 100f0, num_directions=nothing)

Preprocesses the data for SIM analysis by subtracting a background value and normalizing the data.
"""
function preprocess_sim(dat; bg = 100f0, num_directions=nothing, reg_const=2f0, correct_drift=true, enforce_mean=true, verbose=false, damp_edge=true)
    dat = dat .- bg # copies
    # correct slice brightness fluctuations
    dat .*= mean(dat) ./ mean(dat, dims=(1:ndims(dat)-1))

    if (damp_edge)
        # dat .*= window_hanning(size(dat)[1:ndims(dat)-1], border_in=0.97)
        for slice in eachslice(dat, dims=ndims(dat))
            smaller_sz = ceil.(Int, size(slice) .* 0.97)
            rel_sz = (size(slice) .- smaller_sz) ./ smaller_sz
            ns = FindShift.damp_edge_outside(select_region(slice, smaller_sz), rel_sz)
            slice .= ns;
        end
    end

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
                sub_data .*= all_mean .* sub_mean ./ (abs2.(sub_mean) .+ abs2(reg_const))
            end
        end
    end
    return dat
end
