"""
    function estimate_parameters(dat, mypsf=nothing, refdat=nothing; k_vecs=nothing,
                            subtract_mean=true, upsample=false, suppress_sigma=0.0, 
                            num_directions=0, ideal_strength=true, implied_higher_orders=0,
                            otf_exponent = 1.0, otf_moebius = 1.0, amp_magnitudes=nothing, individual_otfs=false, show_quality=true)

Estimate the parameters for a SIM image from the experimatal data. This function is used to estimate the parameters for the SIM image from the experimental data. The function uses the experimental data to estimate the parameters for the SIM image.
The function returns the estimated parameters for the SIM image.

# Arguments
- `dat::Array`: The experimental data. Should be 2-dimensional plus the 3rd dimension for recordings with different phase and/or direction.
- `mypsf::Array`: Optional parameter specifying the PSF to use for prefiltering. If `nothing` is provided, the function will assuma delta-psf.
- `refdat::Array`: The reference data. Typicall the corresponding widefield (zero-order) image.
- `k_vecs::Array`: The k vectors.
- `subtract_mean::Bool`: If true, subtract the mean. Default is true.
- `suppress_sigma`: The width of the center to suppress for the cross-correlation, if > 0. As a ratio of the size. Default is 0. If >0 a suppression of 100% is assumed
- `num_directions`: Number of directions. Default is 0 which means each frame contains all directions. 
                    If provided, it is assumed that the trailing dimension is subdivided into directions and phases per direction.
- `ideal_strength`: If true, the strength of the peaks is set to 1. Default is true.
- `implied_higher_orders`: If not zero, this specifies the number of higher orders which are implied from the first order. Default is 0.
- `amp_magnitudes`: Optional parameter to specify the magnitudes of the amplitudes of the peaks. If `nothing`, the amplitudes are estimated from the data.
- `otf_moebius`: A moebius-polynomial fraction based OTF modification, to use similar as otf_exponent. Default is 1.0.
    It has a different performance particularly at high frequencies suppressing them not as strongly.
- `otf_exponent`: The exponent to use for the OTF. Default is 1.0.
- `individual_otfs`: If true, the function will estimate individual OTFs for each direction. Default is false.
- `show_quality`: If true, the function will print the conditioning quality of the unmixing matrix. Default is true.
- `verbose`: If true, the function will print information about the estimation process. Default is true.
- `phase_only`: If true, the function will only consider the phase information in Fourierspace for the autocorrelation. Default is `false`.
- `peak_ref`: should be `nothing`, except if this was already precomputed using the function `precompute_correlations`.
- `corr_psf`: should be `nothing`, except if this was already precomputed using the function `precompute_correlations`.
- `cropped`: should be `nothing`, except if this was already precomputed using the function `precompute_correlations`.

# Returns
- `sp`: The estimated parameters for the SIM image. This is a `SIMParams` object containing the estimated parameters.


"""
function estimate_parameters(dat, mypsf=nothing, refdat=nothing; k_vecs=nothing, peak_ref=nothing, corr_psf=nothing, cropped=nothing,
                            subtract_mean=true, upsample=false, suppress_sigma=0.15, 
                            num_directions=0, ideal_strength=true, implied_higher_orders=0,
                            otf_exponent = 1.0, otf_moebius = 1.0, amp_magnitudes=nothing, individual_otfs=false,
                            show_quality=true, notch_filter=nothing, verbose=true, phase_only=false, method=:FindIter, scale = 200, roi_size = 2)
    if (num_directions > 0)
        num_phases = size(dat, ndims(dat)) ÷ num_directions;
        if num_phases * num_directions != size(dat, ndims(dat))
            error("The number of phases times the number of directions must equal the number of frames.")
        end
        spf = nothing
        for_print = "("
        for d in 1:num_directions
            sub_data = slice(dat, ndims(dat), (d-1)*num_phases+1:d*num_phases) # pick all phases of one direction
            k_vec = nothing
            if !isnothing(k_vecs)
                k_vec = k_vecs[d]
                if isa(k_vec[1], Number)
                    k_vec = [k_vec,]
                end
            end
            spf_sub = estimate_parameters(sub_data, mypsf, refdat; k_vecs=k_vec,  
                                            subtract_mean=subtract_mean, suppress_sigma=suppress_sigma, 
                                            num_directions=0, ideal_strength=ideal_strength, implied_higher_orders=implied_higher_orders,
                                            amp_magnitudes=amp_magnitudes, otf_exponent=otf_exponent, otf_moebius = otf_moebius, individual_otfs=individual_otfs,
                                            show_quality=show_quality, notch_filter=notch_filter, verbose=verbose, phase_only=phase_only, method=method, scale=scale, roi_size=roi_size)
            if (d == 1)
                spf = spf_sub
            else
                num_orders = size(spf_sub.k_peak_pos, 1)
                if (individual_otfs)
                    maxidx = maximum(spf.otf_indices) 
                    spf.otf_indices = vcat(spf.otf_indices, collect(maxidx+1:maxidx+num_orders-1))
                else
                    spf.otf_indices = vcat(spf.otf_indices, ones(Int, num_orders-1))
                end
                spf.otf_phases = vcat(spf.otf_phases, zeros(num_orders-1))
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
            if (implied_higher_orders != 0)
                for_print *= " $(peaks[1])"
            else
                for_print *= " $(peaks)"
            end
        end
        spf.peak_strengths ./= maximum(spf.peak_strengths)
        if (verbose)
            println("Put the following line in the next call to estimate_parameters: ")
            println("k_vecs =" * for_print * ")")
            if (show_quality)
                println("conditioning quality of unmxing matrix is $(1 / cond(weight_matrix(spf))), (1.0 is best)")
            end
        end
        return spf
    end

    mymean = mean(dat, dims=ndims(dat)) 
    if isnothing(refdat)
        refdat = mymean 
    end
    refdat = squeeze_dim(refdat, ndims(refdat))

    peak_ref, corr_psf, cropped = isnothing(peak_ref) ? precompute_correlations(dat, mypsf; otf_moebius=otf_moebius, otf_exponent=otf_exponent, suppress_sigma=suppress_sigma, subtract_mean=subtract_mean, datmean=mymean) : (peak_ref, corr_psf)

    if isnothing(k_vecs)
        # let the user interactively select the initial k vectors:
        k_vecs, _, _ = get_subpixel_correl(peak_ref; other=refdat, psf=corr_psf, upsample=upsample, correl_mask=nothing, interactive=true, show_quality=show_quality, phase_only=phase_only, method=method, scale=scale, roi_size=roi_size)
        println("You can call this function with the k_vecs parameter $(k_vecs) to speed up the estimation.")
    else
        k_vecs, _, _ = get_subpixel_correl(peak_ref; other=refdat, k_est = k_vecs,  psf=corr_psf, upsample=upsample, correl_mask=nothing, interactive=false, show_quality=show_quality, phase_only=phase_only, method=method, scale=scale, roi_size=roi_size)
    end

    # find_shift(dat[:,:,1], dat[:,:,1])
    # k_vec, phase, amp = get_subpixel_correl(cropped;  psf=psf_cropped, upsample=upsample, k_est=(509, -308))

    k_peak_pos = peak_to_kvecs(k_vecs, size(dat))
    # to_tuple = (t) -> (((2 .* t[1:2] ./ (size(dat)[1:2])...,))..., 0.0)
    # k_peak_pos = [(0.0, 0.0, 0.0), to_tuple.(k_vecs)...]
    # k_peak_pos = [(0.0, 0.0, 0.0), (2 .* k_vecs ./ size(dat))...]
    if (implied_higher_orders > 0)
        base_vec = k_peak_pos[2]
        k_peak_pos = [k_peak_pos[1:2]...]
        for h in 1:implied_higher_orders
            k_peak_pos = vcat(k_peak_pos, base_vec .* (h+1))
        end
    end

    peak_phases = zeros(size(dat, ndims(dat)), length(k_peak_pos))
    peak_strengths = zeros(size(dat, ndims(dat)), length(k_peak_pos))
    for p in axes(cropped, ndims(cropped)) # phases
        # @show p
        cropped_slice = squeeze_dim(slice(cropped, ndims(cropped), p), ndims(cropped))
        rel_corr = get_rel_subpixel_correl(refdat, cropped_slice, k_vecs, corr_psf; upsample=false)
        peak_phases[p, 1] = 0 # peak phase of zero order is always zero
        peak_phases[p, 2:2+length(rel_corr)-1] .= angle.(rel_corr)
        ideal_magnitudes = isnothing(amp_magnitudes) ? 0.5  : order_strengths_from_amp(amp_magnitudes)[2:end]
        peak_strengths[p, 2:end] .= (ideal_strength) ? ideal_magnitudes : abs.(rel_corr)
        if (implied_higher_orders > 0)
            for h in 1:implied_higher_orders
                peak_phases[p, h+2] = angle.(rel_corr[1]) * (h+1)
            end
        end

        peak_strengths[p, 1] = 0.5 # (ideal_strength) ? 1.0 : res_amp # sum(cropped[:,:,p] .* refdat, dims=p) # / prod(size(cropped))
    end

    if (show_quality)
        quality = 1 - abs(sum(cis.(peak_phases[:, 2])))
        println("Phase quality: $(quality)")
    end

    # if (ideal_strength)
    #     peak_strengths = ones(size(peak_strengths)...)
    # end

    otf_indices = let 
        if (individual_otfs)
            collect(1:length(k_peak_pos))
        else
            ones(Int, length(k_peak_pos))
        end
    end
    otf_phases = zeros(length(k_peak_pos))    
    k_peak_pos2 = [d for d in k_peak_pos]
    psfsz = size(dat)[1:ndims(dat)-1]
    mypsf = (isnothing(mypsf)) ? delta(psfsz) : mypsf
    spf = SIMParams(mypsf, k_peak_pos2, peak_phases, peak_strengths, otf_indices, otf_phases, otf_exponent);
    if (show_quality)
        println("conditioning quality of unmxing submatrix is $(1 / cond(weight_matrix(spf))), (1.0 is best)")
    end

    return spf
end

function precompute_correlations(dat, mypsf=nothing; subtract_mean=true, datmean=mean(dat, dims=ndims(dat)), otf_moebius=1f0, otf_exponent=1f0, suppress_sigma=0.15)
    # psf = abs2.(ift(rr(size(dat)[1:2]) .< 0.25*size(dat,1)))
    # psf ./= sum(psf)
    cs = size(dat)[1:2]
    # preprocess the data according to the settings

    cropped = let
        if (subtract_mean)
            # subtract the appropriately scaled mean from each slice.
            Float32.(dat) .- Float32.(datmean) .* sum(dat, dims=(1:ndims(dat)-1)) ./ sum(datmean)
        else
            Float32.(dat)
        end
    end

    # select the PSF to use for prefiltering
    # if isnothing(mypsf)
    #     mypsf = collect(delta(eltype(dat), size(dat)[1:end-1]))
    # end

    corr_psf = mypsf
    if !isnothing(mypsf)
        corr_psf = mypsf ./ sum(mypsf) # let
        corr_otf = fft(corr_psf)
        was_modified = false
        if (otf_moebius != 1 || otf_exponent != 1)
            # gamma = 0.5;x=0:0.01:1; plot(moebius.(x, 1/gamma), label="moebius γ=$(gamma)"); plot!(x.^gamma, label="exp γ=$(gamma)")
            # gamma = 2;x=0:0.01:1; plot!(moebius.(x, 1/gamma), label="moebius γ=$(gamma)"); plot!(x.^gamma, label="exp γ=$(gamma)")
            # gamma = 1;x=0:0.01:1; plot!(moebius.(x, 1/gamma), label="moebius γ=$(gamma)")
            # xlabel!("input"); ylabel!("output"); title!("Moebius vs. Expontial Gamma Correction")
            if (otf_moebius != 1)
                old_mag = abs.(corr_otf);
                gamma = 1/otf_moebius;  # to make it behave like the exponential gamma
                moebius(old_mag, gamma) = gamma*old_mag/(1+(gamma-1)*old_mag)
                corr_otf .= cis.(angle.(corr_otf)) .* moebius.(old_mag, gamma)
            end
            if (otf_exponent != 1)
                new_mag = abs.(corr_otf) .^ otf_exponent
                corr_otf .= cis.(angle.(corr_otf)) .* new_mag
            end
            mypsf = real.(ifft(corr_otf))
            mypsf = mypsf ./ sum(mypsf) # let
            was_modified = true
        end
        # modify the PSF to suppress the low frequencies, if wanted
        if ndims(corr_otf) > 2
            corr_otf = @view corr_otf[:,:,1]
            midz = size(corr_psf,3) ÷ 2 + 1
            corr_psf = @view corr_psf[:,:,midz]
        end
        shift_x = (angle(-corr_otf[2,1])) .* size(corr_otf,1) / 2pi
        shift_y = (angle(-corr_otf[1,2])) .* size(corr_otf,2) / 2pi

        if (abs(shift_x) > 0.05 || abs(shift_y) > 0.05)
            @warn "The PSF is significantly asymmtric or shifted by $(shift_x), $(shift_y).\nThis may lead to problems in the estimation. Trying to correct shift"
            shifter = ifftshift(exp_ikx_col(typeof(corr_otf), size(corr_otf), shift_by=(shift_x, shift_y)))
            corr_otf .*= shifter
            was_modified = true
        end
        if (suppress_sigma > 0)
            # construct a 1-gaussian to suppress the low frequencies of the PSF
            gs = ifftshift(1 .- gaussian_sep(real_arr_type(typeof(corr_psf)), size(corr_psf); sigma=suppress_sigma .* size(corr_psf)))
            corr_otf .*= gs
            was_modified = true
        end
        if (was_modified)
            corr_psf = ifft(corr_otf)
        end
    end

    # use the first provided image as the one to correlate with the reference.
    # The prefiltering is done in get_subpixel_correl.
    peak_ref = squeeze_dim(slice(cropped, ndims(cropped), 1), ndims(cropped)) # [:,:,1]

    return peak_ref, corr_psf, cropped
end

"""
    get_correlation_map(dat, mypsf=nothing; subtract_mean=true, datmean=mean(dat, dims=ndims(dat)), upsample=false)

computes a correlation map that can be used for interactive and automatic peak identification.
# Arguments
- `dat`: The data to correlate
- `psf`: an (optional) psf to determing the weights during correlation. If `nothing` is provided a psf will be assumed.
- `other`: a possibly second dataset to correlate to
- `upsample`: determines whether upsampling (Fourier-padding) is used to calculate the correlation. Upsampling is a little more accurate but slower.
- `subtract_mean`: If true, the mean value will be subtracted

"""
function get_correlation_map(dat, mypsf=nothing, refdat=nothing; subtract_mean=true, datmean=mean(dat, dims=ndims(dat)), upsample=false, otf_moebius=1f0, otf_exponent=1f0, suppress_sigma=0.15)
    mymean = datmean;  
    if isnothing(refdat)
        refdat = mymean 
    end
    refdat = squeeze_dim(refdat, ndims(refdat))

    peak_ref, corr_psf, _ = precompute_correlations(dat, mypsf; subtract_mean=subtract_mean, datmean=mymean, otf_moebius=otf_moebius, otf_exponent=otf_exponent, suppress_sigma=suppress_sigma) 

    dat = peak_ref;
    if isnothing(refdat)
        refdat = dat
    end
    up = FindShift.prepare_correlation(dat, corr_psf; upsample=upsample)
    up_other = FindShift.prepare_correlation(refdat, corr_psf; upsample=upsample)

    ftcorrel = up .* conj.(up_other)

    return ftcorrel, peak_ref, corr_psf
end

"""
    order_strengths_from_amp(amp_magnitudes)

calculates the order_strengths from the magnitudes of the amplitudes using FFTs.

Parameters:
- `amp_magnitudes`: Input magnitudes of the amplitudes at the pupil plane 
"""
function order_strengths_from_amp(amp_magnitudes)
    result_sz = (2*length(amp_magnitudes)-1,)
    res = abs.(fft(abs2.(ifft(ifftshift(select_region(amp_magnitudes,result_sz))))))
    return res[1:length(res)÷2+1]./maximum(res)
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
    to_peak = (t) -> round.(Int, ((t[1:2] .* (sz[1:2])...,) ./ 2))
    return to_peak.(k_vecs)
end

"""
    correlate_raw_data(sim_data, sp::SIMParams)

this function correlates all raw data images and various k-positions according to 
K. Wicker, O. Mandula, G. Best, R. Fiolka, R. Heintzmann, Phase optimisation for structured illumination microscopy, Optics Express 21, 2032–2049, 2013
https://opg.optica.org/oe/fulltext.cfm?uri=oe-21-2-2032
See section 4.3 onwards to calculate the image correlation tensor D_ij^(l) with i, j being the images and l being the shift-k-vector
and from there the optimization can be performed via
C = M^(-1) D^(l) M^(-1)*

with the aim to optimize the phases (and order-strength) to minimize artefacts.
"""
function correlate_raw_data(sim_data, sp::SIMParams)
    num_ks = length(sp.k_peak_pos)
    num_imgs = size(sim_data)[end]
    D_ijl = zeros(ComplexF64, num_imgs, num_imgs, num_ks)
    for (l, myk) in enumerate(sp.k_peak_pos)
        k_exponential = exp_ikx_sep(complex_arr_type(typeof(sim_data)), size(sim_data)[1:end-1]; shift_by=myk)
        for (i, sim_slice1) in enumerate(eachslice(sim_data; dims=ndims(sim_data)))
            for (j, sim_slice2) in enumerate(eachslice(sim_data; dims=ndims(sim_data)))
                D_ijl[i,j,l] = sum(sim_slice1 .* k_exponential .* conj.(sim_slice2))
                D_ijl[j,i,l] = conj.(D_ijl[j,i,l])
            end
        end
    end
    return D_ijl
end

"""
    Cijl_from_Dijl(M, Dijl)

converts the Dijl matrix containing the correlations between images into a 
Cijl matrix, containing the correlations between orders (i.e. k-vector positions).

```jdoctest
Example:
Dijl = StructuredIlluminationMicroscopy.correlate_raw_data(sim_data, sp)
M = StructuredIlluminationMicroscopy.weight_matrix(sp)
StructuredIlluminationMicroscopy.Cijl_from_Dijl(M, Dijl)
```
"""
function Cijl_from_Dijl(M, D_ijl)
    M_inv = pinv(cat(M, conj.(M[:,2:end]), dims=2))
    C_ijl = zeros(size(D_ijl))
    for l = 1:size(D_ijl,3)
        C_ijl[:,:,l] = M_inv*D_ijl[:,:,l]*conj.(M_inv)
    end
    return C_ijl
end
