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

    myinv = haskey(prep,:pinv_weight_mat) ? prep.pinv_weight_mat : pinv_weight_matrix(sp)
    num_orders = size(sp.peak_phases, 2)
    order = haskey(prep,:order) ? prep.order : similar(sim_data, CT, imsz...)
    ftorder = haskey(prep,:ftorder) ? prep.ftorder : similar(sim_data, CT, imsz...)
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
    for n=1:num_orders
        # apply subpixel shifts
        otfmul = haskey(prep, :otfs) ? prep.otfs[sp.otf_indices[n]] : one(RT)
        ordershift = .-sp.k_peak_pos[n] .* expand_size(imsz, ntuple((d)->1, length(sp.k_peak_pos[n]))) ./ 2
        ordershift = ntuple((d)-> (d <= 2) ? ordershift[d] : 0.0, length(ordershift))
        # peakphase = 0.0 # should automatically have been accounted for # .-sp.peak_phases[n, contributing[1]] 
        # peak phases are already accounted for in the weights
        # mydstidx = ntuple(d->(d==ndims(sim_data)) ? n : Colon(), ndims(sim_data))
        dot_mul_last_dim!(order, sim_data, myinv, n); # unmixes (separates) an order from the data, writes into order
        ordershift = shift_subpixel!(order, ordershift, prep, n)
        pixelshifts[n] = ordershift 

        # now place (add) the order with possible weights into the result RFFT image
        # myftorder = ft(order)
        # myftorder .*= otfmul
        ifftshift!(ftorder, order)  # can be saved by incorporating a shift factor into the subpixelshifter
        # ftorder .= order
        prep.plan_fft! * ftorder # fft!
        fftshift!(order, ftorder) 
        order .*= otfmul
        myftorder = order # just an alias
        # @vt myftorder

        # write the order into the result rft image
        select_region!(myftorder, rec; dst_center = bctr .+ ordershift[1:ndims(rec)], operator! = add!)
        # idsbwd = ntuple(d-> (d <= 2) ? (size(myftorder, d):-1:1) : Colon(), ndims(myftorder))
        idsbwd = ntuple(d-> (size(myftorder, d):-1:1), ndims(myftorder))
        bwd_v = @view myftorder[idsbwd...]
        select_region!(bwd_v, rec; dst_center = bctrbwd .- ordershift[1:ndims(rec)], operator! = conj_add!)
    end
    return rec, bsz
end


function get_otfs(ACT, sz, pp::PSFParams, sp::SIMParams, rp, use_rft = false; do_modify=false)
    
    RT = real(eltype(ACT))
    h = psf(sz, pp; sampling=sp.sampling)
    otfs = Array{ACT}(undef, maximum(sp.otf_indices))
    pz = let
        if (length(sz) > 2 && sz[3] > 1)
            zz(RT, (1,1, sz[3]))
        else
            1.0
        end
    end
    hm = h
    for i in 1:length(otfs)
        kz = pi*sp.k_peak_pos[i][3]
        if (kz != 0.0)
            hm = h .* cos.(pz .* kz .+ sp.peak_phases[i])
        end
        myotf = let
            if use_rft
                rfft(ifftshift(hm))
            else
                fftshift(fft(ifftshift(hm)))
            end
        end
        if do_modify
            myotf = modify_otf(myotf, rp.suppression_sigma, rp.suppression_strength)
        end 
        otfs[i] = myotf
    end
    return otfs
end

function recon_sim_prepare(sim_data, pp::PSFParams, sp::SIMParams, rp::ReconParams, do_preallocate=true; use_measure=false, double_use=true)
    sz = size(sim_data)
    imsz = sz[1:end-1]

    # rec = zeros(eltype(sim_data), size(sim_data)[1:end-1])
    RT = eltype(sim_data)
    CT = Complex{RT}
    # construct the modified reconstruction OTF
    ids = ntuple(d->(d==ndims(sim_data)) ? 1 : Colon(), length(sz))
    ACT = typeof(sim_data[ids...] .+ 0im)
    myotfs = get_otfs(ACT, sz[1:end-1], pp, sp, rp, do_modify=true)

    # calculate the pseudo-inverse of the weight-matrix constructed from the information in the SIMParams object
    myinv = pinv_weight_matrix(sp)

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

    prepd = (otfs= myotfs, upsample_factor=rp.upsample_factor, plan_irfft=myplan_irfft, plan_fft! =myplan_fft!, pinv_weight_mat=myinv,
            subpixel_shifters=subpixel_shifters, pixelshifts=pixelshifts)

    if (do_preallocate)
        if (double_use)
            result_rft_tmp = result_rft # since result_rft will now be freshly allocated a little bigger to fit also the final result
            tmp = similar(sim_data, RT, prod(size(result_rft_tmp))*2)
            tmp_cpx = reinterpret(CT, tmp)  # same data, but complex interpretation
            rsize = get_result_size(imsz, rp.upsample_factor)
            result =  reshape(view(tmp,1:prod(rsize)), rsize) # ATTENTION: This reuses the same memory as result_rft !
            result_rft = reshape(tmp_cpx, size(result_rft_tmp)...) # ATTENTION: This reuses the same memory as result_rft !
            if (prod(size(tmp_cpx)) > prod(imsz))
                # GC.@preserve sarr1 begin
                # order = reshape(view(tmp_cpx, 1:prod(imsz)), imsz) # original data size but complex
                # order = unsafe_wrap(ACT, pointer(view(tmp_cpx, 1:prod(imsz))), imsz; own=false)
                # end
                # @show "saved order"
            end
            ftorder = let 
                if (prod(size(tmp_cpx)) >= 2*prod(imsz))
                    # @show "saved ftorder"
                    similar(sim_data, CT, imsz...)
                    # reshape(view(tmp_cpx, prod(imsz)+1:2*prod(imsz)), imsz)  # gives trouble for the in-place fft!, which ignores this reshaped view.
                    # unsafe_wrap(ACT, pointer(view(tmp_cpx, prod(imsz)+1:2*prod(imsz))), imsz; own=false)
                else
                    similar(sim_data, CT, imsz...)
                end
            end

            prepd = (otfs=myotfs, upsample_factor=rp.upsample_factor, plan_irfft=myplan_irfft, plan_fft! =myplan_fft!, pinv_weight_mat=myinv,
                    result_rft=result_rft, result_rft_tmp=result_rft_tmp,
                    order=order, ftorder=ftorder, result=result) # , result_tmp=result_tmp
        else
            result_rft_tmp = get_upsampled_rft(sim_data, prepd)
            rsize = get_result_size(imsz, rp.upsample_factor)
            result =  similar(sim_data, RT, rsize...)
            ftorder =  similar(sim_data, CT, imsz...)
            # result_tmp =  similar(sim_data, RT, get_result_size(imsz, rp.upsample_factor)...)
            prepd = (otfs= myotfs, upsample_factor=rp.upsample_factor, plan_irfft=myplan_irfft, plan_fft! =myplan_fft!, pinv_weight_mat=myinv,
                    result_rft=result_rft, result_rft_tmp=result_rft_tmp, order=order, ftorder=ftorder, result=result) # , result_tmp=result_tmp
        end
    end

    dobj = collect(delta(eltype(sim_data), size(sim_data)[1:end-1]))  # , offset=CtrFFT)
    spd = SIMParams(sp, n_photons = 0.0);
    sim_delta, _ = simulate_sim(dobj, pp, spd);
    ART = typeof(sim_data)
    sim_delta = ART(sim_delta)
    rec_delta = recon_sim(sim_delta, prepd, sp)

    # calculate the final filter
    # rec_otf = ft(rec_delta)
    # rec_otf = fftshift(fft(ifftshift(rec_delta)))
    rec_otf = fftshift(fft(rec_delta))
    rec_otf ./= maximum(abs.(rec_otf))
    # h_goal = RT.(distance_transform(feature_transform(Array(abs.(rec_otf) .< 2e-8))))
    # h_goal ./= maximum(abs.(h_goal))
    h_goal = rec_otf.*0 .+ 1;

    final_filter = ACT(h_goal) .* conj.(rec_otf)./ (abs2.(rec_otf) .+ RT(rp.wiener_eps))
    #final_filter = fftshift(rfft(real.(ifft(ifftshift(final_filter)))), [2,3])
    final_filter = rfftshift(rfft(fftshift(real.(ifft(ifftshift(final_filter))))))

    # the algorithm needs preallocated memory:
    # order: Array{ACT}(undef, size(sim_data)[1:end-1]..., size(sp.peak_phases, 2))
    # result_image: Array{ACT}(undef, size(sim_data)[1:end-1])
    
    if (do_preallocate)
        prep = (otfs = myotfs, upsample_factor=rp.upsample_factor, final_filter=final_filter, plan_irfft=myplan_irfft, plan_fft! =myplan_fft!,
            subpixel_shifters=subpixel_shifters, pixelshifts=pixelshifts, pinv_weight_mat=myinv,
                result_rft=prepd.result_rft, order=prepd.order, ftorder=prepd.ftorder, result=prepd.result, result_rft_tmp=prepd.result_rft_tmp) # result_tmp=prepd.result_tmp, 
    else
        prep = (otfs = myotfs, final_filter=final_filter, 
        subpixel_shifters=subpixel_shifters, pixelshifts=pixelshifts, pinv_weight_mat=myinv,
        upsample_factor=rp.upsample_factor, plan_irfft=myplan_irfft, plan_fft! =myplan_fft!)
    end
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
+ `pp::PSFParams` : PSFParams object
+ `prep::Tuple` : preparation data
+ `sp::SIMParams` : SIMParams object

"""
function recon_sim(sim_data, prep, sp::SIMParams)

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
    # res_tmp = res
    # fftshift_even!(res_tmp, 2:ndims(res_tmp))

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

