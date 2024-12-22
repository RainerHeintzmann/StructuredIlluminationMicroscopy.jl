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
Each separated order is already placed (added) in Fourier space into the result image.

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
        otfmul = prod(size(prep.otfs))>1 ? prep.otfs[n] : one(RT)
        ordershift = .-sp.k_peak_pos[n] .* imsz ./ 2
        ordershift = ntuple((d)-> (d <= 2) ? ordershift[d] : 0.0, length(ordershift))
        # peakphase = 0.0 # should automatically have been accounted for # .-sp.peak_phases[n, contributing[1]] 
        # peak phases are already accounted for in the weights
        dot_mul_last_dim!(order, sim_data, prep.pinv_weight_mat, n); # unmixes (separates) an order from the data, writes into order
        ordershift = shift_subpixel!(order, ordershift, prep, n) # applies a real-space subpixel shift to the order prior to its fft
        pixelshifts[n] = ordershift 

        # now place (add) the order with possible weights into the result RFFT image
        ifftshift!(ftorder, order)  # can be saved by incorporating a shift factor into the subpixelshifter
        prep.plan_fft! * ftorder # fft!
        fftshift!(order, ftorder) 
        if (otfmul != one(RT))
            order .*= otfmul
        end
        myftorder = order # just an alias

        # @show maximum(imag.(order))

        # write the order into the result rft image
        select_region!(myftorder, rec; dst_center = bctr .+ ordershift[1:ndims(rec)], operator! = add!)

        # write the flipped and conjugated order into the result rft image
        idsbwd = ntuple(d-> (size(myftorder, d):-1:1), ndims(myftorder))
        bwd_v = @view myftorder[idsbwd...]
        select_region!(bwd_v, rec; dst_center = bctrbwd .- ordershift[1:ndims(rec)], operator! = conj_add!)
    end
    return rec, bsz
end

"""
    get_otfs(ACT, sz, sp::SIMParams, rp, use_rft = false, mypsf=nothing)

Generate the OTFs for the SIM reconstruction by first simulating a PSF and then (optionally)
creating according to sp.otf_indices the z-modified OTFs for the SIM reconstruction.
The finally returned OTFs correspond to the indices as in `sp.otf_indices`).

Parameters:
+ `ACT` : datatype of the OTFs
+ `sz` : size of the OTFs
+ `sp::SIMParams` : SIMParams object
+ `use_rft` : use RFT instead of FFT

"""
function get_otfs(ACT, sz, sp::SIMParams, use_rft = false; do_modify=false, mypsf=nothing)
    RT = real(eltype(ACT))
    # if isnothing(mypsf)
    #     mypsf = psf(sz, pp; sampling=sp.sampling)
    # end
    mypsf = sp.mypsf
    num_otf_indices = maximum(sp.otf_indices)
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
        if (kz != 0.0)
            hm = mypsf .* cos.(pz .* kz .+ sp.peak_phases[i])
        end
        myotf = (use_rft) ? rfft(ifftshift(hm)) : fftshift(fft(ifftshift(hm)))
        otfs[i] = myotf
    end
    return otfs
end

"""
    get_modified_otfs(ACT, sz, sp::SIMParams, rp, use_rft = false; do_modify=false, mypsf=nothing)

Generate the OTFs for the SIM reconstruction by first simulating a PSF and then (optionally)
creating according to sp.otf_indices the z-modified OTFs for the SIM reconstruction.
The finally returned OTFs correspond to the peak numbering (not the indices in `sp.otf_indices`).

Parameters:
+ `ACT` : datatype of the OTFs
+ `sz` : size of the OTFs
+ `sp::SIMParams` : SIMParams object
+ `rp` : Reconstructions parameters. rp.suppression_sigma and rp.suppression_strength are used to modify the OTFs. rp.preshift_otfs determines whether to preshift the OTFs.
+ `use_rft` : use RFT instead of FFT
+ `do_modify` : modify the OTFs with a suppression filter

"""
function get_modified_otfs(ACT, sz, sp::SIMParams, rp, use_rft = false; do_modify=false, mypsf=nothing)    
    otfs = get_otfs(ACT, sz, sp, use_rft; do_modify=do_modify, mypsf=mypsf)
    if do_modify
        for i in eachindex(otfs)
            otfs[i] = modify_otf(otfs[i], rp.suppression_sigma, rp.suppression_strength) # applies central peak suppression
        end 
    end
    num_otf_indices = length(otfs)
    psfs =  Array{ACT}(undef, num_otf_indices) # only needed for calculating the pre-shifted OTFs
    if (rp.preshift_otfs)
        for i in 1:num_otf_indices
            psfs[i] = (use_rft) ? fftshift(irfft(otfs[i], sz[1])) : fftshift(ifft(ifftshift(otfs[i])))
        end
    end

    all_otfs = Array{ACT}(undef, length(sp.k_peak_pos))
    for i in eachindex(sp.k_peak_pos)
        if (rp.preshift_otfs)
            ordershift = .-sp.k_peak_pos[i] .* expand_size(sz, ntuple((d)->1, length(sp.k_peak_pos[i]))) ./ 2
            myshifter, _ = get_shift_subpixel(psfs[sp.otf_indices[i]], ordershift)
            mypsf = psfs[sp.otf_indices[i]] .* myshifter
            myotf = (use_rft) ? rfft(ifftshift(mypsf)) : fftshift(fft(ifftshift(mypsf)))
            all_otfs[i] = myotf
        else
            all_otfs[i] = otfs[sp.otf_indices[i]]
        end
    end
    
    return all_otfs
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
    recon_sim_prepare(sim_data, sp::SIMParams, rp::ReconParams)

Prepare the SIM reconstruction by generating the OTFs, calculating the pseudo-inverse of the weight matrix,
and preallocating memory for the reconstruction.

Parameters:
+ `sim_data::Array` : simulated SIM data
+ `sp::SIMParams` : SIM configuration parameters 
+ `rp::ReconParams` : Reconstruction parameters. See the help for ReconParams for more information.

"""
function recon_sim_prepare(sim_data, sp::SIMParams, rp::ReconParams; mypsf = nothing)
    ACT = complex_arr_type(typeof(sim_data), Val(ndims(sim_data)-1)) # typeof(sim_data[ids...] .+ 0im)
    ART = real_arr_type(typeof(sim_data), Val(ndims(sim_data)-1)) # typeof(sim_data[ids...] .+ 0im)
    prep = PreparationParams(ART)
    begin 
        sz = size(sim_data)
        imsz = sz[1:end-1]
        if (rp.slice_by_slice && length(imsz)>2 && imsz[3] > 1)
            reference_slice = (rp.reference_slice == 0) ? imsz[3] ÷ 2 + 1 : rp.reference_slice
            sim_data = sim_data[:,:,reference_slice,:]
            prep = recon_sim_prepare(sim_data, pp, sp, rp; mypsf=mypsf)
            return prep
        end

        # rec = zeros(eltype(sim_data), size(sim_data)[1:end-1])
        RT = eltype(sim_data)
        CT = Complex{RT}
        prep.slice_by_slice=rp.slice_by_slice
        prep.upsample_factor = rp.upsample_factor

        # construct the modified reconstruction OTF
        prep.otfs = get_modified_otfs(ACT, sz[1:end-1], sp, rp; do_modify=true, mypsf=mypsf)

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

        # prepd = (otfs= myotfs, upsample_factor=rp.upsample_factor, plan_irfft=myplan_irfft, plan_fft! =myplan_fft!, pinv_weight_mat=myinv,
        #         subpixel_shifters=subpixel_shifters, pixelshifts=pixelshifts, slice_by_slice=rp.slice_by_slice)

        if (rp.do_preallocate)
            pre_allocate!(sim_data, prep, rp)
        end

        dobj = collect(delta(eltype(sim_data), size(sim_data)[1:end-1]))  # , offset=CtrFFT)
        spd = SIMParams(sp, n_photons = 0.0);
        sim_delta, _ = simulate_sim(dobj, spd);
        ART = typeof(sim_data)
        sim_delta = ART(sim_delta)
        rec_delta = recon_sim(sim_delta, prep, sp)

        # calculate the final filter
        # rec_otf = ft(rec_delta)
        # rec_otf = fftshift(fft(ifftshift(rec_delta)))
        rec_otf = fftshift(fft(rec_delta))
        rec_otf ./= maximum(abs.(rec_otf))

        if (rp.use_hgoal)
            h_goal = RT.(distance_transform(feature_transform(Array(abs.(rec_otf) .< rp.hgoal_thresh))))
            h_goal ./= maximum(abs.(h_goal))
            h_goal .^= rp.hgoal_exp
        else
            h_goal = rec_otf.*0 .+ 1;
        end

        prep.final_filter = ACT(h_goal) .* conj.(rec_otf)./ (abs2.(rec_otf) .+ RT(rp.wiener_eps))
        #final_filter = fftshift(rfft(real.(ifft(ifftshift(final_filter)))), [2,3])
        prep.final_filter = rfftshift(rfft(fftshift(real.(ifft(ifftshift(prep.final_filter))))))

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
            rec[:,:,z] = recon_sim(sim_data[:,:,z,:], prep, sp)
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
        mul!(result, prep.plan_irfft, res_tmp)  # to apply an out-of-place irfft with preaccolaed memory
    end
    # rec .= rec_tmp
    # fftshift!(rec, rec_tmp)
    # rec = fftshift(irfft(ifftshift(res, [2,3]), bsz[1]))
    # rec = irft(res, bsz[1]) # real.(ift(res))

    # @vt real.(rec) sum(sim_data, dims=3)[:,:,1]
    # @vt ft(real.(rec)) ft(sum(sim_data, dims=3)[:,:,1]) ft(obj)

    return result
end

