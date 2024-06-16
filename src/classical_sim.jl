function get_upsampled_rft(sim_data, prep::NamedTuple)
    if haskey(prep, :result_rft)
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

    otfmul = haskey(prep, :otf) ? prep.otf : one(RT)

    myinv = haskey(prep,:pinv_weight_mat) ? prep.pinv_weight_mat : pinv_weight_matrix(sp)
    num_orders = size(sp.peak_phases, 2)
    order = haskey(prep,:order) ? prep.order : similar(sim_data, CT, imsz...)
    ftorder = haskey(prep,:ftorder) ? prep.ftorder : similar(sim_data, CT, imsz...)
    sz = size(order)
    bsz = ceil.(Int, sz .* prep.upsample_factor) # backwards size
    rec = get_upsampled_rft(sim_data, prep) # gets the memory and clears the result array
    # define the center coordinate of the rft result rec
    bctr = get_rft_center(bsz)
    # define a shifted center coordinate to account for the flip of even sizes, when appying a flip operation
    bctrbwd = bctr .+ iseven.(bsz)

    pixelshifts = Array{NTuple{3, Int}}(undef, num_orders)
    # if (otfmul <: AbstractArray)
    #     otfmul = ifftshift(otfmul)
    # end
    for n=1:num_orders
        # apply subpixel shifts
        ordershift = .-sp.k_peak_pos[n] .* expand_size(imsz, ntuple((d)->1, length(sp.k_peak_pos[n]))) ./ 2
        # peakphase = 0.0 # should automatically have been accounted for # .-sp.peak_phases[n, contributing[1]] 
        # peak phases are already accounted for in the weights
        # mydstidx = ntuple(d->(d==ndims(sim_data)) ? n : Colon(), ndims(sim_data))
        dot_mul_last_dim!(order, sim_data, myinv, n); # unmixes (separates) an order from the data, writes into order
        ordershift = shift_subpixel!(order, ordershift, prep, n)
        pixelshifts[n] = ordershift 

        # now place (add) the order with possible weights into the result RFFT image
        # myftorder = ft(order)
        # myftorder .*= otfmul
        ifftshift!(ftorder, order)
        # ftorder .= order
        prep.plan_fft!*ftorder # fft!
        fftshift!(order, ftorder) 
        order .*= otfmul
        myftorder = order # just an alias
        # @vt myftorder

        select_region!(myftorder, rec; dst_center = bctr .+ ordershift[1:ndims(rec)], operator! = add!)
        idsbwd = ntuple(d-> (size(myftorder, d):-1:1), ndims(myftorder))
        bwd_v = @view myftorder[idsbwd...]
        select_region!(bwd_v, rec; dst_center = bctrbwd .- ordershift[1:ndims(rec)], operator! = conj_add!)
    end
    return rec, bsz
end

function recon_sim_prepare(sim_data, pp::PSFParams, sp::SIMParams, rp::ReconParams, do_preallocate=true; use_measure=false)
    sz = size(sim_data)
    imsz = sz[1:end-1]
    h = psf(sz[1:end-1], pp; sampling=sp.sampling)
    # rec = zeros(eltype(sim_data), size(sim_data)[1:end-1])
    RT = eltype(sim_data)
    CT = Complex{RT}
    # construct the modified reconstruction OTF
    myotf = modify_otf(fftshift(fft(ifftshift(h))), RT(rp.suppression_sigma), RT(rp.suppression_strength))
    ids = ntuple(d->(d==ndims(sim_data)) ? 1 : Colon(), ndims(sim_data))

    # calculate the pseudo-inverse of the weight-matrix constructed from the information in the SIMParams object
    myinv = pinv_weight_matrix(sp)

    ACT = typeof(sim_data[ids...] .+ 0im)
    myotf = ACT(myotf)
    result_rft = get_upsampled_rft(sim_data, (upsample_factor=rp.upsample_factor,))
    myplan_irfft = (use_measure) ? plan_irfft(result_rft, get_result_size(imsz, rp.upsample_factor)[1], flags=FFTW.MEASURE) : plan_irfft(result_rft, get_result_size(imsz, rp.upsample_factor)[1])
    order =  similar(sim_data, CT, imsz...)
    myplan_fft! = (use_measure) ? plan_fft!(order, flags=FFTW.MEASURE) : plan_fft!(order)

    num_orders = size(sp.peak_phases,2)
    pixelshifts = Array{NTuple{3, Int}}(undef, num_orders)
    subpixel_shifters = Vector{Any}(undef, num_orders)
    for n in 1:num_orders
        ordershift = .-sp.k_peak_pos[n] .* expand_size(imsz, ntuple((d)->1, length(sp.k_peak_pos[n]))) ./ 2
        # ordershift = shift_subpixel!(order, ordershift, prep, n)
        subpixel_shifters[n], pixelshifts[n] = get_shift_subpixel(order, ordershift)
    end

    prepd = (otf= myotf, upsample_factor=rp.upsample_factor, plan_irfft=myplan_irfft, plan_fft! =myplan_fft!, pinv_weight_mat=myinv,
            subpixel_shifters=subpixel_shifters, pixelshifts=pixelshifts)

    if (do_preallocate)
        result_rft_tmp = get_upsampled_rft(sim_data, prepd)
        rsize = get_result_size(imsz, rp.upsample_factor)
        result =  similar(sim_data, RT, rsize...)
        # tmp_view = @view reinterpret(RT, result_rft)[1:prod(rsize)]
        # result = reshape(tmp_view, rsize)  # ATTENTION: This reuses the same memory as result_rft !
        # valid_rng = ntuple(d->1:rsize[d], length(rsize))
        # result = @view reinterpret(RT, result_rft)[valid_rng...]  # ATTENTION: This reuses the same memory as result_rft !
        # result_tmp =  similar(sim_data, RT, get_result_size(imsz, rp.upsample_factor)...)
        ftorder =  similar(sim_data, CT, imsz...)
        prepd = (otf= myotf, upsample_factor=rp.upsample_factor, plan_irfft=myplan_irfft, plan_fft! =myplan_fft!, pinv_weight_mat=myinv,
                result_rft=result_rft, result_rft_tmp=result_rft_tmp, order=order, ftorder=ftorder, result=result) # , result_tmp=result_tmp
    end

    dobj = delta(eltype(sim_data), size(sim_data)[1:end-1])  # , offset=CtrFFT)
    spd = SIMParams(sp, n_photons = 0.0);
    sim_delta, _ = simulate_sim(dobj, pp, spd);
    ART = typeof(sim_data)
    sim_delta = ART(sim_delta)
    rec_delta = recon_sim(sim_delta, prepd, sp, rp)

    # calculate the final filter
    # rec_otf = ft(rec_delta)
    # rec_otf = fftshift(fft(ifftshift(rec_delta)))
    rec_otf = fftshift(fft(rec_delta))
    rec_otf ./= maximum(abs.(rec_otf))
    h_goal = RT.(distance_transform(feature_transform(Array(abs.(rec_otf) .< 0.0002))))
    h_goal ./= maximum(abs.(h_goal))

    final_filter = ACT(h_goal) .* conj.(rec_otf)./ (abs2.(rec_otf) .+ RT(rp.wiener_eps))
    #final_filter = fftshift(rfft(real.(ifft(ifftshift(final_filter)))), [2,3])
    final_filter = rfftshift(rfft(fftshift(real.(ifft(ifftshift(final_filter))))))

    # the algorithm needs preallocated memory:
    # order: Array{ACT}(undef, size(sim_data)[1:end-1]..., size(sp.peak_phases, 2))
    # result_image: Array{ACT}(undef, size(sim_data)[1:end-1])
    
    if (do_preallocate)
        prep = (otf= myotf, upsample_factor=rp.upsample_factor, final_filter=final_filter, plan_irfft=myplan_irfft, plan_fft! =myplan_fft!,
            subpixel_shifters=subpixel_shifters, pixelshifts=pixelshifts, pinv_weight_mat=myinv,
                result_rft=prepd.result_rft, order=prepd.order, ftorder=prepd.ftorder, result=prepd.result, result_rft_tmp=prepd.result_rft_tmp) # result_tmp=prepd.result_tmp, 
    else
        prep = (otf= myotf, final_filter=final_filter, 
        subpixel_shifters=subpixel_shifters, pixelshifts=pixelshifts, pinv_weight_mat=myinv,
        upsample_factor=rp.upsample_factor, plan_irfft=myplan_irfft, plan_fft! =myplan_fft!)
    end
    return prep
end

"""
    recon_sim(sim_data, prep, sp::SIMParams, rp::ReconParams)

performs a classical SIM reconstruction. 
1) Order separation and applying subpixel-shifts.
2) RFT of each separated order
3) Multiplication of each FT-order with frequency-dependent strength and order phase
4) Fourier-placement of the orders and upsampling and summation into final ft-image.
5) IFT of the final ft-image

Parameters:
+ `sim_data::Array` : simulated SIM data
+ `pp::PSFParams` : PSFParams object
+ `prep::Tuple` : preparation data
+ `sp::SIMParams` : SIMParams object
+ `rp::ReconParams` : ReconParams object

"""
function recon_sim(sim_data, prep, sp::SIMParams, rp::ReconParams)

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
    if (haskey(prep, :final_filter))
        res .*= prep.final_filter
    end

    # apply IFT of the final ft-image

    res_tmp = haskey(prep, :result_rft_tmp) ? prep.result_rft_tmp : similar(res);
    result = haskey(prep, :result) ? prep.result : similar(sim_data, eltype(sim_data), bsz...)
    # rec_tmp = haskey(prep, :result_tmp) ? prep.result_tmp : similar(rec)

    rifftshift!(res_tmp, res)
    if isnothing(prep.plan_irfft)
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

