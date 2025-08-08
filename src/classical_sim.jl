function get_upsampled_rft(sim_data, prep::PreparationParams)
    if prod(size(prep.result_rft)) > 1
        res = prep.result_rft
        res .= zero(complex(eltype(sim_data)))
    else
        sz = size(sim_data)[1:end-1]
        bsz = get_result_size(sz, prep.upsample_factor)
        res = similar(sim_data, complex(eltype(sim_data)), rfft_size(bsz)...)
        res .= zero(complex(eltype(sim_data)))
    end
    return res # since this represents an rft
end

"""
    separate_and_place_orders(sim_data, sp::SIMParams), prep)

Separate the orders in the SIM data and apply subpixel shifts in real space. 
Each separated order is immediately placed (added) in Fourier space into the final result image.

Parameters:
+ `sim_data::Array` : simulated SIM data
+ `sp::SIMParams` : SIMParams object

"""
function separate_and_place_orders(sim_data, sp::SIMParams, prep)
    RT = eltype(sim_data)
    CT = Complex{RT}
    imsz = size(sim_data)[1:end-1]

    prep.pinv_weight_mat = prod(size(prep.pinv_weight_mat))>1 ? prep.pinv_weight_mat : pinv_weight_matrix(sp)
    num_orders = size(sp.peak_phases, 2)
    order = prod(size(prep.order))>1 ? prep.order : similar(sim_data, CT, imsz...)
    ftorder = prod(size(prep.ftorder))>1 ? prep.ftorder : similar(sim_data, CT, imsz...)
    sz = size(order)
    upsample_factor = ntuple((d) -> (d<=2) ? prep.upsample_factor : 1, length(sz))
    bsz = ceil.(Int, sz .* upsample_factor) # backwards size
    rec = get_upsampled_rft(sim_data, prep) # gets the memory and clears the result array
    # define the center coordinate of the rft result rec
    bctr = get_rft_center(bsz)
    # define a shifted center coordinate to account for the flip of even sizes, when appying a flip operation
    bctrbwd = bctr .+ iseven.(bsz)

    pixelshifts = Array{NTuple{3, Int}}(undef, num_orders)
    # if (otfmul <: AbstractArray)
    #     otfmul = ifftshift(otfmul)
    # end
    imsz = expand_size(imsz, ntuple((d)->1, length(sp.k_peak_pos[1]))) 
    for n=1:num_orders
        # apply subpixel shifts
        ordershift = .-sp.k_peak_pos[n] .* imsz ./ 2
        ordershift = ntuple((d)-> (d <= 2) ? ordershift[d] : 0.0, length(ordershift))
        # peakphase = 0.0 # should automatically have been accounted for # .-sp.peak_phases[n, contributing[1]] 
        # peak phases are already accounted for in the weights
        # unmix (separate) an order from the data, writes into order:
        dot_mul_last_dim!(order, sim_data, prep.pinv_weight_mat, n);
        # apply a real-space subpixel shift to the order prior to its fft:
        ordershift = shift_subpixel!(order, ordershift, prep, n) 
        pixelshifts[n] = ordershift 

        # now place (add) the order with possible weights into the result RFFT image
        # ifftshift can be avoided in the future by incorporating a shift factor into the subpixelshifter:
        ifftshift!(ftorder, order)
        prep.plan_fft! * ftorder # fft!
        fftshift!(order, ftorder) 
        # if an otf-modification is provided it is multiplied here:
        if (prod(size(prep.otfs))>1)
            order .*= prep.otfs[n]
        end
        myftorder = order # just an alias

        # @show maximum(imag.(order))

        # write the order into the result rft image
        select_region!(myftorder, rec; dst_center = bctr .+ ordershift[1:ndims(rec)], operator! = add!)

        # write the flipped and conjugated order into the result rft image. Except for the zero order, which is written only once
        if (n>1)
            idsbwd = ntuple(d-> (size(myftorder, d):-1:1), ndims(myftorder))
            bwd_v = @view myftorder[idsbwd...]
            select_region!(bwd_v, rec; dst_center = bctrbwd .- ordershift[1:ndims(rec)], operator! = conj_add!)
        end
    end
    return rec, bsz
end

"""
    get_otfs(ACT, sz, sp::SIMParams, use_rft = false, slice_by_slice=false)

Generate the OTFs for the SIM reconstruction from PSFs provided in `sp.mypsf` and then (optionally)
creating according to sp.otf_indices the z-modified OTFs for the SIM reconstruction.
The finally returned OTFs correspond to the indices as in `sp.otf_indices`).

Parameters:
+ `ACT` : datatype of the OTF arrays
+ `sz` : size of the OTFs
+ `sp::SIMParams` : SIMParams object
+ `use_rft` : use RFT instead of FFT

"""
function get_otfs(ACT, sz, sp::SIMParams, use_rft = false)
    RT = real(eltype(ACT))
    mypsf = sp.mypsf
    if (ndims(ACT) < 3 && size(mypsf,3)>1)
        mypsf = @view mypsf[:,:,size(mypsf,3) ÷ 2 +1]
    end
    num_otf_indices = maximum(sp.otf_indices)
    # @show ACT
    # @show slice_by_slice
    # @show size(mypsf)
    otfs = Array{ACT}(undef, num_otf_indices)
    pz = let
        if (length(sz) > 2 && sz[3] > 1)
            zz(RT, (1,1, sz[3]))
        else
            1.0
        end
    end
    hm = mypsf
    for i in eachindex(otfs)
        kz = pi*sp.k_peak_pos[i][3]
        if (kz != 0.0) # shift to +/- kz position and account for the z-related misadjustment via peak_phase:
            hm = mypsf .* cos.(pz .* kz .+ sp.peak_phases[i])
        end
        myotf = (use_rft) ? rfft(ifftshift(hm)) : fftshift(fft(ifftshift(hm)))
        otfs[i] = myotf
    end
    return otfs
end

function get_otf_mask(myotf, otf_radius=0.0; otf_threshold=0.001, rel_blur=0.1)
    mask = let 
        if otf_radius > 0.0
            window_radial_hanning(size(myotf), border_in = otf_radius.-rel_blur, border_out = otf_radius)
        else
            abs.(myotf) .> maximum(abs.(myotf)) * otf_threshold  # mask to avoid division by zero
        end
    end
    return mask
end

"""
    get_modified_otfs(ACT, sz, sp::SIMParams, rp, use_rft = false; do_modify=false)

Generate the OTFs for the SIM reconstruction by first simulating a PSF and then (optionally)
creating according to sp.otf_indices the z-modified OTFs for the SIM reconstruction.
The finally returned OTFs correspond to the peak numbering (not the indices in `sp.otf_indices`).

Parameters:
+ `ACT` : datatype of the OTFs
+ `sz` : size of the OTFs
+ `sp::SIMParams` : SIMParams object
+ `rp` : Reconstructions parameters. rp.suppression_sigma and rp.suppression_strength are used to modify the OTFs. rp.preshift_otfs determines whether to preshift the OTFs.
+ `rp.notch` : a notch filter to apply to the OTFs, if nothing is given, a Gaussian notch filter is used. The notch can also be a Vector of individual notch filters, one for each OTF
+ `use_rft` : use RFT instead of FFT
+ `do_modify` : modify the OTFs with an additional suppression filter
+ `otf_threshold` : threshold to avoid division by zero in the Wiener filter, default is 0.002

"""
function get_modified_otfs(ACT, sz, sp::SIMParams, rp, use_rft = false; do_modify=false, otf_threshold=0.001)
    otfs = get_otfs(ACT, sz, sp, use_rft)
    # @show order_vars =  get_sep_order_variances(sp)    
    # otfs ./= sqrt.(order_vars) # square the OTFs to account for the weighting with the inverse variance after compensation of the OTF
    if (do_modify == false) # just to see if this flag can be removed in the future
        @warn "DO_MODIFY is false"
    end

    # for (otf) in otfs
       # otf .*= otf .* sqrt(order_var) # square the OTFs to account for the weighting with the inverse variance after compensation of the OTF
       #  otf .= otf .* sqrt(order_var) # square the OTFs to account for the weighting with the inverse variance after compensation of the OTF
       # otf .*= otf  # square the OTFs to account for the weighting with the inverse variance after compensation of the OTF
    # end
    if do_modify
        my_notch = isnothing(rp.notch) ? gaussian_notch(first(otfs), rp.suppression_strength, rp.suppression_sigma) : rp.notch;
        if (!isnothing(rp.notch) && rp.suppression_strength > 0.0)
            @warn "An rp.notch filter was explicitly provided, but rp.suppression_strength of $(rp.suppression_strength) was also provided, but is ignored."
        end
        for i in eachindex(otfs)              
            if ndims(my_notch) == ndims(otfs[i])+1 
                otfs[i] .*= my_notch[:,:,i]
            else
                if size(my_notch, 2) == 1
                    error("The notch filter must be an array or a vector of length $(length(otfs)), but has length $(length(my_notch)).")
                end
                otfs[i] .*= my_notch
            end
        end 
    end
    num_otf_indices = length(otfs)
    psfs =  Array{ACT}(undef, num_otf_indices) # only needed for calculating the pre-shifted OTFs
    if (rp.preshift_otfs)
        for i in 1:num_otf_indices
            psfs[i] = (use_rft) ? fftshift(irfft(otfs[i], sz[1])) : fftshift(ifft(ifftshift(otfs[i])))
        end
    end

    all_weights = Array{ACT}(undef, length(sp.k_peak_pos))
    ART = real_arr_type(ACT) 
    all_masks = Array{ART}(undef, length(sp.k_peak_pos))

    for i in eachindex(sp.k_peak_pos)
        if (rp.preshift_otfs)
            ordershift = .-sp.k_peak_pos[i] .* expand_size(sz, ntuple((d)->1, length(sp.k_peak_pos[i]))) ./ 2
            myshifter, _ = get_shift_subpixel(psfs[sp.otf_indices[i]], ordershift)
            mypsf = psfs[sp.otf_indices[i]] .* myshifter
            myotf = (use_rft) ? rfft(ifftshift(mypsf)) : fftshift(fft(ifftshift(mypsf)))
            all_weights[i] = myotf  
            all_masks[i] = get_otf_mask(myotf, rp.otf_radius; otf_threshold=otf_threshold) # mask to avoid division by zero
        else
            myotf = otfs[sp.otf_indices[i]]
            all_weights[i] = myotf 
            all_masks[i] = get_otf_mask(myotf, rp.otf_radius; otf_threshold=otf_threshold)
        end
    end

    # otf_masks = [abs.(otf) .> maximum(abs.(otf)) * otf_threshold for otf in otfs] # mask to avoid division by zero
    
    return all_weights, all_masks
end


function pre_allocate!(sim_data, prep, rp)
    RT = eltype(sim_data)
    CT = Complex{RT}
    sz = size(sim_data)
    imsz = sz[1:end-1]

    if (rp.double_use)
        prep.result_rft_tmp = prep.result_rft # since result_rft will now be freshly allocated a little bigger to fit also the final result
        tmp = similar(sim_data, RT, prod(size(prep.result_rft_tmp))*2)
        tmp_cpx = reinterpret(CT, tmp)  # same data, but complex interpretation
        rsize = get_result_size(imsz, rp.upsample_factor)
        prep.result =  reshape(view(tmp,1:prod(rsize)), rsize) # ATTENTION: This reuses the same memory as result_rft !
        prep.result_rft = reshape(tmp_cpx, size(prep.result_rft_tmp)...) # ATTENTION: This reuses the same memory as result_rft !
        if (prod(size(tmp_cpx)) > prod(imsz))
            # GC.@preserve sarr1 begin
            # order = reshape(view(tmp_cpx, 1:prod(imsz)), imsz) # original data size but complex
            # order = unsafe_wrap(ACT, pointer(view(tmp_cpx, 1:prod(imsz))), imsz; own=false)
            # end
            # @show "saved order"
        end
        prep.ftorder = let 
            if (prod(size(tmp_cpx)) >= 2*prod(imsz))
                # @show "saved ftorder"
                similar(sim_data, CT, imsz...)
                # reshape(view(tmp_cpx, prod(imsz)+1:2*prod(imsz)), imsz)  # gives trouble for the in-place fft!, which ignores this reshaped view.
                # unsafe_wrap(ACT, pointer(view(tmp_cpx, prod(imsz)+1:2*prod(imsz))), imsz; own=false)
            else
                similar(sim_data, CT, imsz...)
            end
        end

        # prepd = (otfs=myotfs, upsample_factor=rp.upsample_factor, plan_irfft=myplan_irfft, plan_fft! =myplan_fft!, pinv_weight_mat=myinv,
        #         result_rft=result_rft, result_rft_tmp=result_rft_tmp,
        #         order=order, ftorder=ftorder, result=result, slice_by_slice=rp.slice_by_slice) # , result_tmp=result_tmp
    else
        prep.result_rft_tmp = get_upsampled_rft(sim_data, prep)
        rsize = get_result_size(imsz, rp.upsample_factor)
        prep.result =  similar(sim_data, RT, rsize...)
        prep.ftorder =  similar(sim_data, CT, imsz...)
        # result_tmp =  similar(sim_data, RT, get_result_size(imsz, rp.upsample_factor)...)
        # prepd = (otfs= myotfs, upsample_factor=rp.upsample_factor, plan_irfft=myplan_irfft, plan_fft! =myplan_fft!, pinv_weight_mat=myinv,
        #         result_rft=result_rft, result_rft_tmp=result_rft_tmp, order=order, ftorder=ftorder, result=result,
        #         slice_by_slice=rp.slice_by_slice) # , result_tmp=result_tmp
    end
end

"""
    normalize_otfs(otfs keep_hf = false, otf_thresh = 0.002)

Each OTF is normalized such that the sum of abs square values after shifting the other OTFs amounts to one.
This guarantees that the noise will still be flat after applying these OTFs as filters (k-dependent weights) to each separated order.

Parameters:
+ `all_otfs::Array` : array of OTFs
+ `shift_vecs::Array` : array of shift vectors for each OTF
+ `keep_hf::Bool` : whether to keep the high-frequency components of the OTFs (default: false)

"""
function normalize_otfs!(prep; keep_hf = false, otf_masks=nothing, otf_thresh = 0.004)
    if isnothing(otf_masks)
        otf_masks = [get_otf_mask(myotf, rp.otf_radius; otf_threshold=otf_thresh) for myotf in prep.otfs]
    end
    augment_otf(otf, otf_mask) = ifelse(keep_hf, otf .* otf_mask .+ (1 .- otf_mask) .* (maximum(abs.(otf)) / 1000), otf .* otf_mask) # will the automatically be contributing only in regions of single otfs
    # augment_otf(otf, otf_mask) = ifelse(keep_hf, otf .+ (maximum(abs.(otf)) / 1000), otf .* otf_mask) # will the automatically be contributing only in regions of single otfs
    @assert length(prep.otfs) == length(otf_masks) "The number of otfs and otf_masks must be equal, but got $(length(prep.otfs)) and $(length(otf_masks))."

    for (otf, otf_mask) in zip(prep.otfs, otf_masks)
        otf .= augment_otf(otf, otf_mask) # will the automatically be contributing only in regions of single otfs
    end

    all_psfs = [fftshift(ifft(ifftshift(otf))) for otf in prep.otfs] # the psfs are used to calculate the subpixel shifts, so they need to be in real space;
    sum_otfs2 = similar(prep.otfs[1]) # abs2.(otf)
    sqr(x) = x .* x
    for (otf, otf_mask, otf_num) in zip(prep.otfs, otf_masks, eachindex(prep.otfs))
        sum_otfs2 .= eltype(otf)(0) # zero the sum
        mid_pos = size(otf) .÷ 2 .+ 1
        ref_shifter = prep.subpixel_shifters[otf_num]
        for (mypsf, psf_num) in zip(all_psfs, eachindex(all_psfs))
            rel_shift = prep.pixelshifts[otf_num] .- prep.pixelshifts[psf_num] 
            sum_otfs2 .+= select_region(sqr.(fftshift(fft(ifftshift(conj.(prep.subpixel_shifters[psf_num]) .* ref_shifter .* mypsf)))); center = mid_pos .+ rel_shift[1:length(mid_pos)])
            # sum_otfs2 .+= select_region(sqr.(fftshift(fft(ifftshift((conj.(mypsf)))))); center = mid_pos .+ rel_shift[1:length(mid_pos)])
            # the central order is only added once and also the noise scales differently without a shift            
            if (psf_num != otf_num) # (norm(rel_shift) > 0)
                rel_shift = prep.pixelshifts[otf_num] .+ prep.pixelshifts[psf_num]
                sum_otfs2 .+= select_region(sqr.(fftshift(fft(ifftshift((prep.subpixel_shifters[psf_num] .* ref_shifter .* conj.(mypsf)))))); center = mid_pos .+ rel_shift[1:length(mid_pos)])
                # sum_otfs2 .+= select_region(sqr.(fftshift(fft(ifftshift(mypsf)))); center = mid_pos .+ rel_shift[1:length(mid_pos)])
            end
            # sum_otfs2 .+= select_region(abs2.(fftshift(fft(ifftshift(mypsf)))); center = mid_pos .+ rel_shift[1:length(mid_pos)])
        end
        if (keep_hf)
            # not clear, why this augementation here is bad:
            # otf = augment_otf(otf, otf_mask) # keep the high-frequency components

            # sum_otfs_mask = abs.(sum_otfs2) .> maximum(abs.(sum_otfs2)) / 1000
            # otf[sum_otfs_mask] ./= sqrt.(sum_otfs2[sum_otfs_mask]) # guarantees the noise to be constant
            # otf[.~sum_otfs_mask] .= 0 # guarantees the noise to be constant
            otf ./= sqrt.(sum_otfs2) # guarantees the noise to be constant
            # otf .= sum_otfs_mask
        else
            om = otf_mask .> 1e-5
            otf[om] ./= sqrt.(sum_otfs2[om]) # guarantees the noise to be constant
            otf[.~om] .= 0 # guarantees the noise to be constant
        end
        # rel_shift = prep.pixelshifts[otf_num] .+ prep.pixelshifts[1]
        # otf .= select_region(sqr.(fftshift(fft(ifftshift((conj.(all_psfs[1])))))); center = mid_pos .+ rel_shift[1:length(mid_pos)])
        #otf .= sum_otfs2
    end
end

function test_unmix_real(sim_data, sp::SIMParams)
    pinv_weight_mat = pinv_weight_matrix(sp)
    num_orders = size(sp.k_peak_pos, 1)
    orders = [];
    for n=1:num_orders
        order = similar(sim_data, Complex{eltype(sim_data)}, size(sim_data)[1:end-1]...)    
        dot_mul_last_dim!(order, sim_data, pinv_weight_mat, n);
        push!(orders, order)
    end
    return orders
end


"""
    recon_sim_prepare(sim_data, sp::SIMParams, rp::ReconParams)

Prepare the SIM reconstruction by generating the OTFs, calculating the pseudo-inverse of the weight matrix,
and preallocating memory for the reconstruction.

Parameters:
+ `sim_data::Array` : simulated SIM data
+ `sp::SIMParams` : SIM configuration parameters 
+ `rp::ReconParams` : Reconstruction parameters. See the help for ReconParams for more information.

"""
function recon_sim_prepare(sim_data, sp::SIMParams, rp::ReconParams; use_final_filter=true)
    ACT = complex_arr_type(typeof(sim_data), Val(ndims(sim_data)-1)) # typeof(sim_data[ids...] .+ 0im)
    ART = real_arr_type(typeof(sim_data), Val(ndims(sim_data)-1)) # typeof(sim_data[ids...] .+ 0im)
    prep = PreparationParams(ART)
    begin 
        sz = size(sim_data)
        imsz = sz[1:end-1]
        if (rp.slice_by_slice && length(imsz)>2 && imsz[3] > 1)
            reference_slice = (rp.reference_slice == 0) ? imsz[3] ÷ 2 + 1 : rp.reference_slice
            sim_data = sim_data[:,:,reference_slice,:]
            prep = recon_sim_prepare(sim_data, sp, rp)
            return prep
        end

        # rec = zeros(eltype(sim_data), size(sim_data)[1:end-1])
        RT = eltype(sim_data)
        CT = Complex{RT}
        prep.slice_by_slice=rp.slice_by_slice
        prep.upsample_factor = rp.upsample_factor

        # construct the modified reconstruction OTF
        prep.otfs, otf_masks = get_modified_otfs(ACT, sz[1:end-1], sp, rp; do_modify=true)

        # calculate the pseudo-inverse of the weight-matrix constructed from the information in the SIMParams object
        prep.pinv_weight_mat = pinv_weight_matrix(sp)

        prep.result_rft = get_upsampled_rft(sim_data, prep)
        prep.plan_irfft = (rp.use_measure) ? plan_irfft(prep.result_rft, get_result_size(imsz, rp.upsample_factor)[1], flags=FFTW.MEASURE) : plan_irfft(prep.result_rft, get_result_size(imsz, rp.upsample_factor)[1])
        prep.order =  similar(sim_data, CT, imsz...)
        prep.plan_fft! = (rp.use_measure) ? plan_fft!(prep.order, flags=FFTW.MEASURE) : plan_fft!(prep.order)

        num_orders = size(sp.peak_phases,2)
        prep.pixelshifts = Array{NTuple{3, Int}}(undef, num_orders)
        prep.subpixel_shifters = Vector{Any}(undef, num_orders)

        for n in 1:num_orders
            ordershift = .-sp.k_peak_pos[n] .* expand_size(imsz, ntuple((d)->1, length(sp.k_peak_pos[n]))) ./ 2
            # ordershift = shift_subpixel!(order, ordershift, prep, n)
            prep.subpixel_shifters[n], prep.pixelshifts[n] = get_shift_subpixel(prep.order, ordershift)
        end
        normalize_otfs!(prep; otf_masks=otf_masks, keep_hf = rp.keep_hf)
        # @show order_vars =  get_sep_order_variances(sp)    
        # prep.otfs ./= sqrt.(order_vars) # square the OTFs to account for the weighting with the inverse variance after compensation of the OTF

        # prepd = (otfs= myotfs, upsample_factor=rp.upsample_factor, plan_irfft=myplan_irfft, plan_fft! =myplan_fft!, pinv_weight_mat=myinv,
        #         subpixel_shifters=subpixel_shifters, pixelshifts=pixelshifts, slice_by_slice=rp.slice_by_slice)

        if (rp.do_preallocate)
            pre_allocate!(sim_data, prep, rp)
        end

        dobj = collect(delta(eltype(sim_data), size(sim_data)[1:end-1]))  # , offset=CtrFFT)
        spd = SIMParams(sp, n_photons = 0);
        # simulate the noise-free sim data of a single delta peak to obtain the SIM PSF to be used for constructing the inverse.
        sim_delta, _ = simulate_sim(dobj, spd);
        ART = typeof(sim_data)
        sim_delta = ART(sim_delta)

        rec_delta = recon_sim(sim_delta, prep, sp)

        # calculate the final filter
        # rec_otf = ft(rec_delta)
        # rec_otf = fftshift(fft(ifftshift(rec_delta)))
        rec_otf = fftshift(fft(rec_delta))
        rec_otf ./= maximum(abs.(rec_otf))

        rrel = RT.(distance_transform(feature_transform(Array(abs.(rec_otf) .< rp.hgoal_thresh))))
        rrel .= 1 .- rrel/maximum(abs.(rrel))
        h_goal = rp.hgoal.(rrel)

        # only save this rec_otf if it is used for deconvolution
        if (rp.do_deconvolve)
            prep.rec_otf = rec_otf
            prep.deconv_lambda = rp.deconv_lambda
        end

        if (use_final_filter)
            if (rp.keep_hf)
                # modified Wiener filter to allow preserving high-frequency components:
                # prep.final_filter = ACT(h_goal) .* (conj.(rec_otf)  .+ RT(rp.wiener_eps))./ (abs2.(rec_otf) .+ RT(rp.wiener_eps))
                prep.final_filter = ACT(h_goal) .* conj.(rec_otf)./ (abs2.(rec_otf) .+ RT(rp.wiener_eps))
            else
                # standard Wiener filter:
                prep.final_filter = ACT(h_goal) .* conj.(rec_otf)./ (abs2.(rec_otf) .+ RT(rp.wiener_eps))
            end
            #final_filter = fftshift(rfft(real.(ifft(ifftshift(final_filter)))), [2,3])
            prep.final_filter = rfftshift(rfft(fftshift(real.(ifft(ifftshift(prep.final_filter))))))
            # prep.final_filter = (conj.(rec_otf) .+ RT(rp.wiener_eps) ) ./ (abs2.(rec_otf) .+ RT(rp.wiener_eps))
        else
            prep.final_filter = ACT(ones(size(prep.result_rft))) # no final filter
        end

        # the algorithm needs preallocated memory:
        # order: Array{ACT}(undef, size(sim_data)[1:end-1]..., size(sp.peak_phases, 2))
        # result_image: Array{ACT}(undef, size(sim_data)[1:end-1])
        
        # if (rp.do_preallocate)
        #     prep = (otfs = myotfs, upsample_factor=rp.upsample_factor, final_filter=final_filter, plan_irfft=myplan_irfft, plan_fft! =myplan_fft!,
        #         subpixel_shifters=subpixel_shifters, pixelshifts=pixelshifts, pinv_weight_mat=myinv,
        #             result_rft=prepd.result_rft, order=prepd.order, ftorder=prepd.ftorder, result=prepd.result,
        #             result_rft_tmp=prepd.result_rft_tmp, slice_by_slice=rp.slice_by_slice) # result_tmp=prepd.result_tmp, 
        # else
        #     prep = (otfs = myotfs, final_filter=final_filter, 
        #     subpixel_shifters=subpixel_shifters, pixelshifts=pixelshifts, pinv_weight_mat=myinv,
        #     upsample_factor=rp.upsample_factor, plan_irfft=myplan_irfft, plan_fft! =myplan_fft!, slice_by_slice=rp.slice_by_slice)
        # end
    end

    GC.gc();
    return prep
end

"""
    recon_sim(sim_data, prep, sp::SIMParams)

performs a classical SIM reconstruction. Note that the reconstruction parameter set is not needed,
since this information is contained in the `prep` named Tuple information.

1) Order separation and applying subpixel-shifts.
2) RFT of each separated order
3) Multiplication of each FT-order with frequency-dependent strength and order phase
4) Fourier-placement of the orders and upsampling and summation into final ft-image.
5) IFT of the final ft-image

Parameters:
+ `sim_data::Array` : simulated SIM data
+ `prep::Tuple` : preparation data
+ `sp::SIMParams` : SIMParams object

"""
function recon_sim(sim_data, prep, sp::SIMParams)
    sz = size(sim_data)
    imsz = sz[1:end-1]

    # use slice-by-slice reconstruction, if requested and data is not 2D
    if (prep.slice_by_slice && length(imsz)>2 && imsz[3] > 1)
        rec = similar(sim_data, eltype(sim_data), (size(prep.result)..., size(sim_data,3)))
        for z in axes(sim_data, 3)
            rec[:,:,z] .= recon_sim(sim_data[:,:,z,:], prep, sp)
        end
        return rec
    end

    # first separate (unmix) the orders in real space. This also sets the correct global phase.
    # and apply subpixel shifts 
    # orders, pixelshifts = separate_orders(sim_data, sp)
    # apply FFT
    # and perform Fourier-placement of the orders and upsampling and summation into final ft-image
    # res, bsz = place_orders_upsample(orders, pixelshifts, rp.upsample_factor, prep.otf)

    # apply FFT
    # and perform Fourier-placement of the orders and upsampling and summation into final ft-image
    res, bsz = separate_and_place_orders(sim_data, sp, prep)
    
    # apply final frequency-dependent multiplication (filtering)
    if (prod(size(prep.rec_otf))>1) # use deconvolution
        # just for deconvolution we need the psf first
        mypsf = real.(ifft(ifftshift(prep.rec_otf)))
        res_tmp = copy(res)
        rifftshift!(res_tmp, res)
        result = prod(size(prep.result))>1 ? prep.result : similar(sim_data, eltype(sim_data), bsz...)
        result .= irfft(res_tmp,  bsz[1])       # fftshift(irfft(res_tmp,  bsz[1])) 
        iterations = 8
        @info "deconvolving with rec_otf, iterations: $iterations"
        # result, o = deconvolution(result, mypsf, regularizer=TH(), λ=0.0001, loss=Anscombe(100f0));
        # result, o = deconvolution(result, mypsf, regularizer=TH(), mapping=nothing, λ=prep.deconv_lambda, loss=Gauss());
        result, o = deconvolution(result, mypsf, regularizer=TH(), λ=prep.deconv_lambda, loss=Gauss(), iterations=iterations);
        return result
    else
        if (prod(size(prep.final_filter))>1) # haskey(prep, :final_filter))
            res .*= prep.final_filter
        end
        # apply IFT of the final ft-image

        res_tmp = prod(size(prep.result_rft_tmp))>1 ? prep.result_rft_tmp : similar(res);
        result = prod(size(prep.result))>1 ? prep.result : similar(sim_data, eltype(sim_data), bsz...)
        # rec_tmp = haskey(prep, :result_tmp) ? prep.result_tmp : similar(rec)
        rifftshift!(res_tmp, res)
        # res_tmp = res
        # fftshift_even!(res_tmp, 2:ndims(res_tmp))

        if (prod(size(prep.plan_irfft))>=1) # isnothing(prep.plan_irfft)
            result .= irfft(res_tmp,  bsz[1])
        else
            # apply an out-of-place irfft with pre-accolaed memory
            mul!(result, prep.plan_irfft, res_tmp)
        end
        # once can omit the final fftshift, which will be autocompensated by the phases in Fourierspace obtaind from the delta simulation and reconstruction
        return fftshift(result)
    end

    # rec .= rec_tmp
    # fftshift!(rec, rec_tmp)
    # rec = fftshift(irfft(ifftshift(res, [2,3]), bsz[1]))
    # rec = irft(res, bsz[1]) # real.(ift(res))

    # @vt real.(rec) sum(sim_data, dims=3)[:,:,1]
    # @vt ft(real.(rec)) ft(sum(sim_data, dims=3)[:,:,1]) ft(obj)

end

