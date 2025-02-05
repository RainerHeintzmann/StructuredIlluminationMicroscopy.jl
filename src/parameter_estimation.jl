
"""
    estimate_parameters(dat, mypsf=nothing, refdat=nothing; mypsf=nothing, subtract_mean=true,  k_vecs=nothing)

Estimate the parameters for a SIM image from the experimatal data. This function is used to estimate the parameters for the SIM image from the experimental data. The function uses the experimental data to estimate the parameters for the SIM image.
The function returns the estimated parameters for the SIM image.

# Arguments
- `dat::Array`: The experimental data. Should be 2-dimensional plus the 3rd dimension for recordings with different phase and/or direction.
- `mypsf::Array`: Optional parameter specifying the PSF to use for prefiltering. If `nothing` is provided, the function will assuma delta-psf.
- `refdat::Array`: The reference data. Typicall the corresponding widefield (zero-order) image.
- `subtract_mean::Bool`: Subtract the mean.
- `k_vecs::Array`: The k vectors.
- `subtract_mean::Bool`: If true, subtract the mean. Default is true.
- `suppress_sigma`: The width of the center to suppress if > 0. As a ratio of the size. Default is 0.
- 'num_directions': Number of directions. Default is 0 which means each frame contains all directions. 
                    If provided, it is assumed that the trailing dimension is subdivided into directions and phases per direction.
- 'ideal_strength': If true, the strength of the peaks is set to 1. Default is true.
- 'imply_higher_orders': If not zero, this specifies the number of higher orders which are implied from the first order. Default is 0.

"""
function estimate_parameters(dat, mypsf=nothing, refdat=nothing; k_vecs=nothing,
                            subtract_mean=true, upsample=false, suppress_sigma=0.0, 
                            num_directions=0, ideal_strength=true, imply_higher_orders=0)
    if num_directions > 0
        num_phases = size(dat, ndims(dat)) ÷ num_directions;
        if num_phases * num_directions != size(dat, ndims(dat))
            error("The number of phases times the number of directions must equal the number of frames.")
        end
        spf = nothing
        for_print = "("
        for d in 1:num_directions
            sub_data = slice(dat, ndims(dat), (d-1)*num_phases+1:d*num_phases)
            k_vec = nothing
            if !isnothing(k_vecs)
                k_vec = k_vecs[d]
                if isa(k_vec[1], Number)
                    k_vec = [k_vec,]
                end
            end
            spf_sub = estimate_parameters(sub_data, mypsf, refdat; k_vecs=k_vec,  
                                            subtract_mean=subtract_mean, suppress_sigma=suppress_sigma, 
                                            num_directions=0, ideal_strength=ideal_strength, imply_higher_orders=imply_higher_orders)
            if (d == 1)
                spf = spf_sub
            else
                num_orders = size(spf_sub.k_peak_pos, 1)
                spf.otf_indices = vcat(spf.otf_indices, ones(Int, num_orders-1))
                spf.otf_phases = vcat(spf.otf_phases, ones(num_orders-1))
                spf.k_peak_pos = vcat(spf.k_peak_pos, spf_sub.k_peak_pos[2:end])
                spf.peak_phases = hcat(spf.peak_phases, zeros(size(spf.peak_phases, 1), num_orders-1))
                spf.peak_phases = vcat(spf.peak_phases, zeros(num_phases, size(spf.peak_phases,2)))
                spf.peak_phases[end-num_phases+1:end, end-(num_orders-2):end] = spf_sub.peak_phases[:,2:end]

                spf.peak_strengths = hcat(spf.peak_strengths, zeros(size(spf.peak_strengths, 1), num_orders-1))
                spf.peak_strengths = vcat(spf.peak_strengths, zeros(num_phases, size(spf.peak_strengths,2)))
                spf.peak_strengths[end-num_phases+1:end, end-(num_orders-2):end] = spf_sub.peak_strengths[:,2:end]
                spf.peak_strengths[end-num_phases+1:end, 1] .= spf_sub.peak_strengths[:,1]
            end
            if (d>1)
                for_print *= ","
            end

            peaks = kvecs_to_peak(spf_sub.k_peak_pos[2:end], size(dat))
            if (imply_higher_orders != 0)
                for_print *= " $(peaks[1])"
            else
                for_print *= " $(peaks)"
            end
        end
        spf.peak_strengths ./= maximum(spf.peak_strengths)
        println("Put the following line in the next call to estimate_parameters: ")
        println("k_vecs =" * for_print * ")")
        return spf
    end
    # psf = abs2.(ift(rr(size(dat)[1:2]) .< 0.25*size(dat,1)))
    # psf ./= sum(psf)
    cs = size(dat)[1:2]
    # preprocess the data according to the settings
    mymean = mean(dat, dims=ndims(dat)) 
    cropped = let
        if (subtract_mean)
            if isnothing(refdat)
                refdat = mymean 
            end
            Float32.(dat) .- Float32.(mymean) .* sum(dat, dims=(1:ndims(dat)-1)) ./ sum(mymean)
        else
            if isnothing(refdat)
                refdat = mymean
            end
            Float32.(dat)
        end
    end

    # select the PSF to use for prefiltering
    corr_psf = mypsf ./ sum(mypsf) # let

    # modify the PSF to suppress the low frequencies, if wanted
    if !isnothing(corr_psf) && (suppress_sigma > 0)
        # construct a 1-gaussian to suppress the low frequencies of the PSF
        gs = ifftshift(1 .- gaussian_sep(real_arr_type(typeof(corr_psf)), size(corr_psf); sigma=suppress_sigma .* size(corr_psf)))
        corr_psf = ifft(fft(corr_psf) .* gs)
    end

    # use the first provided image as the one to correlate with the reference.
    # The prefiltering is done in get_subpixel_correl.
    peak_ref = @view cropped[:,:,1]

    if isnothing(k_vecs)
        k_vecs, _, _ = get_subpixel_correl(peak_ref; other=refdat, psf=corr_psf, upsample=upsample, correl_mask=nothing, interactive=true)
        println("You can call this function with the k_vecs parameter $(k_vecs) to speed up the estimation.")
    else
        k_vecs, _, _ = get_subpixel_correl(peak_ref; other=refdat, k_est = k_vecs,  psf=corr_psf, upsample=upsample, correl_mask=nothing, interactive=false)
    end

    # find_shift(dat[:,:,1], dat[:,:,1])
    # k_vec, phase, amp = get_subpixel_correl(cropped;  psf=psf_cropped, upsample=upsample, k_est=(509, -308))

    k_peak_pos = peak_to_kvecs(k_vecs, size(dat))
    # to_tuple = (t) -> (((2 .* t[1:2] ./ (size(dat)[1:2])...,))..., 0.0)
    # k_peak_pos = [(0.0, 0.0, 0.0), to_tuple.(k_vecs)...]
    # k_peak_pos = [(0.0, 0.0, 0.0), (2 .* k_vecs ./ size(dat))...]
    if (imply_higher_orders > 0)
        base_vec = k_peak_pos[2]
        k_peak_pos = [k_peak_pos[1:2]...]
        for h in 1:imply_higher_orders
            k_peak_pos = vcat(k_peak_pos, base_vec .* (h+1))
        end
    end

    peak_phases = zeros(size(dat, 3), length(k_peak_pos))
    peak_strengths = zeros(size(dat, 3), length(k_peak_pos))
    for p in axes(cropped, ndims(cropped)) # phases
        rel_corr = get_rel_subpixel_correl(refdat, cropped[:,:,p], k_vecs, corr_psf; upsample=false)
        peak_phases[p, 1] = 0 # peak phase of zero order is always zero
        peak_phases[p, 2:2+length(rel_corr)-1] .= angle.(rel_corr)
        peak_strengths[p, 2:end] .= (ideal_strength) ? 0.5 : abs.(rel_corr)
        if (imply_higher_orders > 0)
            for h in 1:imply_higher_orders
                peak_phases[p, h+2] = angle.(rel_corr[1]) * (h+1)
            end
        end

        peak_strengths[p, 1] = 0.5 # (ideal_strength) ? 1.0 : res_amp # sum(cropped[:,:,p] .* refdat, dims=p) # / prod(size(cropped))
    end

    # if (ideal_strength)
    #     peak_strengths = ones(size(peak_strengths)...)
    # end

    num_photons = 0.0 # ignore this
    bg_photons = 0.0
    otf_indices = ones(Int, length(k_peak_pos))
    otf_phases = zeros(length(k_peak_pos))
    k_peak_pos2 = [d for d in k_peak_pos]
    psfsz = size(dat)[1:ndims(dat)-1]
    mypsf = (isnothing(mypsf)) ? delta(psfsz) : mypsf
    spf = SIMParams(mypsf, num_photons, bg_photons, k_peak_pos2, peak_phases, peak_strengths, otf_indices, otf_phases);
    return spf
end


"""
    peak_to_kvecs(peak_pos; sz)

Convert the peak positions to k vectors.

# Arguments
- `peak_pos::Array`: The peak positions.
- `sz::Tuple`: The size of the image.
"""
function peak_to_kvecs(peak_pos, sz)
    to_tuple = (t) -> (((2 .* t[1:2] ./ (sz[1:2])...,))..., 0.0)
    return [(0.0, 0.0, 0.0), to_tuple.(peak_pos)...]
end

"""
    kvecs_to_peak(k_vecs; sz)

Convert the k vectors to peak positions.

# Arguments
- `k_vecs::Array`: The k vectors.
- `sz::Tuple`: The size of the image.
"""
function kvecs_to_peak(k_vecs, sz)
    to_peak = (t) -> round.(Int, (((t[1:2] .* (sz[1:2])...,) ./ 2)..., 0.0))
    return to_peak.(k_vecs)
end

