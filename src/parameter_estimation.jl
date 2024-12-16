
"""
    estimate_parameters(dat, refdat=nothing; mypsf=nothing, subtract_mean=true, pp::PSFParams=PSFParams(), sampling=(0.11, 0.11, 0.1), prefilter_correl=true, k_vecs=nothing)

Estimate the parameters for a SIM image from the experimatal data. This function is used to estimate the parameters for the SIM image from the experimental data. The function uses the experimental data to estimate the parameters for the SIM image.
The function returns the estimated parameters for the SIM image.

# Arguments
- `dat::Array`: The experimental data.
- `refdat::Array`: The reference data.
- `mypsf::Array`: Optional parameter specifying the PSF to use for prefiltering.
- `subtract_mean::Bool`: Subtract the mean.
- `pp::PSFParams`: The PSF parameters, which are used to generate the PSF, if the PSF is not provided.
- `sampling::Tuple`: The sampling. Default is (0.1, 0.1, 0.1), each in µm.
- `k_vecs::Array`: The k vectors.
- `prefilter_correl::Bool`: If true, prefilter the correlation using the PSF. Default is true. 0.05 may be a useful value.
- `subtract_mean::Bool`: If true, subtract the mean. Default is true.
- `suppress_sigma::Int`: The width of the center to suppress if > 0. As a ratio of the size. Default is 0.

"""
function estimate_parameters(dat, refdat=nothing; pp::PSFParams, k_vecs=nothing, mypsf=nothing,
                            prefilter_correl=true, subtract_mean=true, suppress_sigma=0.0, sampling=(0.1, 0.1, 0.1))
    # psf = abs2.(ift(rr(size(dat)[1:2]) .< 0.25*size(dat,1)))
    # psf ./= sum(psf)
    cs = size(dat)[1:2]
    cropped = let
        if (subtract_mean)
            mymean = mean(dat, dims=ndims(dat)) 
            if isnothing(refdat)
                refdat = mymean 
            end
            Float32.(dat) .- Float32.(mymean) .* sum(dat[:,:,1]) ./ sum(mymean)
        else
            if isnothing(refdat)
                mymean = mean(dat, dims=ndims(dat))
                refdat = mymean
            end
            Float32.(dat)
        end
    end
    peak_ref = cropped[:,:,1]
    corr_psf = let
        if (prefilter_correl)
            if isnothing(mypsf)
                psf(cs, pp; sampling=sampling)
            else
                mypsf
            end
        else
            nothing
        end
    end
    if !isnothing(corr_psf) && (suppress_sigma > 0)
        # construct a 1-gaussian to suppress the low frequencies of the PSF
        gs = ifftshift(1 .- gaussian_sep(real_arr_type(typeof(corr_psf)), size(corr_psf); sigma=suppress_sigma .* size(corr_psf)))
        corr_psf = ifft(fft(corr_psf) .* gs)
    end
    if isnothing(k_vecs)
        k_vecs, _, _ = get_subpixel_correl(peak_ref; other=refdat, psf=corr_psf, upsample=false, correl_mask=nothing, interactive=true)
    else
        k_vecs, _, _ = get_subpixel_correl(peak_ref; other=refdat, k_est = k_vecs,  psf=corr_psf, upsample=false, correl_mask=nothing, interactive=false)
    end
    # find_shift(dat[:,:,1], dat[:,:,1])
    # k_vec, phase, amp = get_subpixel_correl(cropped;  psf=psf_cropped, upsample=false, k_est=(509, -308))

    # @show k_vecs
    to_tuple = (t) -> (((2 .* t[1:2] ./ (size(dat)[1:2])...,))..., 0.0)
    k_peak_pos = [(0.0, 0.0, 0.0), to_tuple.(k_vecs)...]
    # k_peak_pos = [(0.0, 0.0, 0.0), (2 .* k_vecs ./ size(dat))...]

    # @show length(k_peak_pos)
    peak_phases = zeros(size(dat, 3), length(k_peak_pos))
    peak_strengths = zeros(size(dat, 3), length(k_peak_pos))
    for p in axes(cropped, ndims(cropped))
        _, res_phase, res_amp = get_subpixel_correl(cropped[:,:,p];  method = :FindPhase, psf=corr_psf, upsample=false, correl_mask=nothing, interactive=false, k_est=k_vecs)
        peak_phases[p, 1] = 0
        peak_phases[p, 2:end] = res_phase
        peak_strengths[p, 2:end] = res_amp
        peak_strengths[p, 1] = sum(cropped) / sqrt(prod(size(cropped)))
    end
    peak_strengths = peak_strengths ./ maximum(peak_strengths[:,2:end])
    peak_strengths = ones(size(peak_strengths)...)

    num_photons = 0.0 # ignore this
    bg_photons = 0.0
    otf_indices = ones(Int, length(k_peak_pos))
    otf_phases = zeros(length(k_peak_pos))
    k_peak_pos2 = [d for d in k_peak_pos]
    spf = SIMParams(pp, sampling, num_photons, bg_photons, k_peak_pos2, peak_phases, peak_strengths, otf_indices, otf_phases);
    return spf
end