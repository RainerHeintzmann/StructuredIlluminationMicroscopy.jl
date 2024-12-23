
"""
    estimate_parameters(dat, mypsf=nothing, refdat=nothing; mypsf=nothing, subtract_mean=true, prefilter_correl=true, k_vecs=nothing)

Estimate the parameters for a SIM image from the experimatal data. This function is used to estimate the parameters for the SIM image from the experimental data. The function uses the experimental data to estimate the parameters for the SIM image.
The function returns the estimated parameters for the SIM image.

# Arguments
- `dat::Array`: The experimental data.
- `mypsf::Array`: Optional parameter specifying the PSF to use for prefiltering.
- `refdat::Array`: The reference data.
- `subtract_mean::Bool`: Subtract the mean.
- `k_vecs::Array`: The k vectors.
- `prefilter_correl::Bool`: If true, prefilter the correlation using the PSF. Default is true. 0.05 may be a useful value.
- `subtract_mean::Bool`: If true, subtract the mean. Default is true.
- `suppress_sigma`: The width of the center to suppress if > 0. As a ratio of the size. Default is 0.
- 'num_directions': Number of directions. Default is 0 which means each frame contains all directions. 
                    If provided, it is assumed that the trailing dimension is subdivided into directions and phases per direction.

"""
function estimate_parameters(dat, mypsf=nothing, refdat=nothing; k_vecs=nothing,
                            prefilter_correl=true, subtract_mean=true, upsample=false, suppress_sigma=0.0, 
                            num_directions=0, ideal_strength=true)
    if num_directions > 0
        num_phases = size(dat, ndims(dat)) ÷ num_directions;
        if num_phases * num_directions != size(dat, ndims(dat))
            error("The number of phases times the number of directions must equal the number of frames.")
        end
        spf = nothing
        for d in 1:num_directions
            sub_data = slice(dat, ndims(dat), (d-1)*num_phases+1:d*num_phases)
            k_vec = nothing
            if !isnothing(k_vecs)
                k_vec = [k_vecs[d]]
            end
            spf_sub = estimate_parameters(sub_data, mypsf, refdat; k_vecs=k_vec,  
                                            prefilter_correl=prefilter_correl, 
                                            subtract_mean=subtract_mean, suppress_sigma=suppress_sigma, 
                                            num_directions=0, ideal_strength=ideal_strength)
            if (d == 1)
                spf = spf_sub
            else
                num_orders = size(spf_sub.k_peak_pos, 1)
                spf.otf_indices = vcat(spf.otf_indices, 1)
                spf.otf_phases = vcat(spf.otf_phases, 1)
                spf.k_peak_pos = vcat(spf.k_peak_pos, spf_sub.k_peak_pos[2:end])
                spf.peak_phases = hcat(spf.peak_phases, zeros(size(spf.peak_phases, 1), num_orders-1))
                spf.peak_phases = vcat(spf.peak_phases, zeros(num_phases, size(spf.peak_phases,2)))
                spf.peak_phases[end-num_phases+1:end, end-(num_orders-2):end] = spf_sub.peak_phases[:,2:end]

                spf.peak_strengths = hcat(spf.peak_strengths, zeros(size(spf.peak_strengths, 1), num_orders-1))
                spf.peak_strengths = vcat(spf.peak_strengths, zeros(num_phases, size(spf.peak_strengths,2)))
                spf.peak_strengths[end-num_phases+1:end, end-(num_orders-2):end] = spf_sub.peak_strengths[:,2:end]
                spf.peak_strengths[end-num_phases+1:end, 1] .= spf_sub.peak_strengths[:,1]
            end
        end
        spf.peak_strengths ./= maximum(spf.peak_strengths)
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
    #     if (prefilter_correl)
    #         if isnothing(mypsf)
    #             psf(cs, pp; sampling=sampling)
    #         else
    #             mypsf
    #         end
    #     else
    #         nothing
    #     end
    # end

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

    to_tuple = (t) -> (((2 .* t[1:2] ./ (size(dat)[1:2])...,))..., 0.0)
    k_peak_pos = [(0.0, 0.0, 0.0), to_tuple.(k_vecs)...]
    # k_peak_pos = [(0.0, 0.0, 0.0), (2 .* k_vecs ./ size(dat))...]

    peak_phases = zeros(size(dat, 3), length(k_peak_pos))
    peak_strengths = zeros(size(dat, 3), length(k_peak_pos))
    for p in axes(cropped, ndims(cropped)) # phases
        rel_corr = get_rel_subpixel_correl(refdat, cropped[:,:,p], k_vecs, corr_psf; upsample=false)
        peak_phases[p, 1] = 0 # peak phase of zero order is always zero
        peak_phases[p, 2:end] .= angle.(rel_corr)
        peak_strengths[p, 2:end] .= (ideal_strength) ? 0.5 : abs.(rel_corr)

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