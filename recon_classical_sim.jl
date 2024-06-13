using View5D
using FFTW
# using FourierTools
using IndexFunArrays # for rr and delta
using PointSpreadFunctions # to calculate PSFs
using TestImages
using LinearAlgebra # for dot
using NDTools  # select_region!
using Noise # for poisson simulation
using Images # for distance transform
using BenchmarkTools
using CUDA
using SeparableFunctions

mutable struct SIMParams
    psf_params::PSFParams
    sampling::NTuple{3, Float64}
    n_photons::Float64
    n_photons_bg::Float64

    k_peak_pos::Array{NTuple{3, Float64}, 1}  # peak-positions in k-space, a vector of 3D tuples

    # peak-phases in k-space. This is a 2D array with the first dimension being the number of peaks
    # and the second dimension being the number of phases  (i.e. the phases in each image)
    peak_phases::Array{Float64,2}
    # peak-intensities in k-space. This is a 2D array with the first dimension being the number of peaks
    # and the second dimension being the number of intensities   (i.e. the intensities of each peak in each image)
    # peak_strengths being zero are simply skipped in the calculation
    peak_strengths::Array{Float64,2}

    function SIMParams(psf_params::PSFParams, sampling::NTuple{3, Float64}, n_photons::Float64, n_photons_bg::Float64, k_peak_pos::Array{NTuple{3, Float64}, 1}, peak_phases::Array{Float64,2}, peak_strengths::Array{Float64,2})
        new(psf_params, sampling, n_photons, n_photons_bg, k_peak_pos, peak_phases, peak_strengths)
    end
    function SIMParams(sp::SIMParams; sampling=sp.sampling, psf_params=sp.psf_params, n_photons=sp.n_photons, n_photons_bg=sp.n_photons_bg, k_peak_pos=sp.k_peak_pos, peak_phases=sp.peak_phases, peak_strengths=sp.peak_strengths)
        new(psf_params, sampling, n_photons, n_photons_bg, k_peak_pos, peak_phases, peak_strengths)
    end
end

mutable struct ReconParams
    suppression_sigma::Float64
    otf_mul::Float64
    upsample_factor::Int
    wiener_eps::Float64

    function ReconParams(suppression_sigma::Float64 = 0.2, otf_mul::Float64 = 1.0, upsample_factor::Int = 2, wiener_eps::Float64 = 1e-6)    
        new(suppression_sigma, otf_mul, upsample_factor, wiener_eps)
    end
end

"""
    SIMPattern(p, sp.peak_pos, sp.peak_phases, sp.peak_strengths)

Generate the SIM illumination pattern.
Parameters:
+ `h::PSF` : PSF object
+ sp::SIMParams : SIMParams object

"""
function SIMPattern(h, sp::SIMParams, n)
    sim_pattern = zeros(eltype(h), size(h))
    pos = idx(eltype(h), size(h))
    for i in 1:size(sp.k_peak_pos, 1)
        k = pi.*sp.k_peak_pos[i][1:ndims(h)] # sp.k_peak_pos is relative to the Nyquist frequency of the image
        if (sp.peak_strengths[n,i] != 0.0)
            if (norm(k) == 0.0)
                sim_pattern .+= sp.peak_strengths[n,i]
            else
                sim_pattern .+= cos.(dot.(Ref(k), pos) .+ sp.peak_phases[n,i]) .* sp.peak_strengths[n,i]
            end
        end
    end
    return sim_pattern
end

"""
    generate_peaks(num_phases::Int=3, num_directions::Int=3, num_orders=0.0, k0 = 0.9)

generates peaks assuming grating-like illuminations
Parameters:
+ `num_phases::Int` : number of phases
+ `num_directions::Int` : number of directions
+ `num_orders::Int` : number of orders
+ `k0::Float64` : peak frequency (relative to the Nyquist frequency of the image)
"""
function generate_peaks(num_phases::Int=9, num_directions::Int=3, num_orders::Int=2, k0 = 0.9)
    num_peaks = num_directions * num_orders;
    k_peak_pos = Array{NTuple{3, Float64}}(undef, num_peaks)
    for d in 1:num_directions
        for o in 1:num_orders
            theta = 2pi*(d-1)/num_directions
            k = k0 .* (o-1)
            if (k == 0.0)
                k_peak_pos[(d-1)*num_orders + o] = (0.0, 0.0, 0.0);
            else
                k_peak_pos[(d-1)*num_orders + o] = k .* (cos(theta), sin(theta), 0.0)
            end
        end
    end
    peak_phases = zeros(num_phases, num_peaks)
    peak_strengths = zeros(num_phases, num_peaks)
    phases_per_cycle = num_phases ÷ num_directions
    for p in 1:num_phases
        d = (p-1) ÷ phases_per_cycle  # directions start at 0
        for o in 1:num_orders
            peak_phases[p, d*num_orders + o] = 2pi*(o-1)*(p-1)/phases_per_cycle
            peak_strengths[p, d*num_orders + o] = 1.0
        end
    end
    return k_peak_pos, peak_phases, peak_strengths
end

"""
    simulate_sim(obj, pp::PSFParams, sp::SIMParams)

Simulate SIM data.
Parameters:
+ `obj::Array` : object
+ `pp::PSFParams` : PSFParams object
+ `sp::SIMParams` : SIMParams object
+ `sampling::Tuple` : sampling in µm
"""
function simulate_sim(obj, pp::PSFParams, sp::SIMParams)
    sz = size(obj)
    h = psf(sz, pp; sampling=sp.sampling);  # sampling is 0.5 x 0.5 x 2.0 µm

    # sim_data = Array{eltype(obj)}(undef, size(h)..., size(sp.peak_phases, 1))
    sim_data = similar(obj, eltype(obj), size(h)..., size(sp.peak_phases, 1))
    # Generate SIM illumination pattern
    otf = rfft(ifftshift(h))
    for n in 1:size(sp.peak_phases, 1)
        sim_pattern = SIMPattern(h, sp, n)
        myidx = ntuple(d->(d==ndims(sim_data)) ? n : Colon(), ndims(sim_data))
        # sim_data[myidx...] .= conv_psf(obj .* sim_pattern, h)
        sim_data[myidx...] .= irfft(rfft(obj .* sim_pattern) .*otf, size(h,1))
    end

    if (sp.n_photons != 0.0)
        sim_data .*= sp.n_photons ./ maximum(sim_data)
        sim_data .= eltype(sim_data).(poisson(Float64.(sim_data))) # cast is due to a bug in the poisson function
    end
    return sim_data
end

function weight_matrix(sp)
    return cis.(sp.peak_phases) .* sp.peak_strengths
end

function pinv_weight_matrix(sp; Eps=1e-6)
    res = pinv(weight_matrix(sp))
    res[abs.(res) .< Eps] .= 0.0
    return res
end

"""
    shift_subpixel!(img, ordershift, peakphase)

Shifts the Fourier-transform of the image by subpixel values and returns the pixelshift.
"""
function shift_subpixel!(img, ordershift, prep, order_num)
    # pos = idx(eltype(img), size(img))
    # img .*= cis.(dot.(Ref(2pi .* subpixelshift ./ size(img)), pos))
    subpixel_shifters, pixelshift = (haskey(prep, :subpixelshifts)) ? (prep.subpixel_shifters[order_num], prep.pixelshift[order_num]) : get_shift_subpixel(img, ordershift)
    img .*= subpixel_shifters
    return pixelshift
end

function get_shift_subpixel(img, subpixelshift)
    pixelshift = round.(Int, ordershift)
    subpixelshift = ordershift[1:2] .- pixelshift[1:2]
    if (norm(subpixelshift) == 0.0)
        return pixelshift
    end
    return exp_ikx_sep(typeof(img), size(img); shift_by=.-subpixelshift), pixelshift
end

"""
    dot_mul_last_dim!(orders, sim_data, myinv, n, Eps = 1e-7)

performs the dot product of the last dimension of the SIM data with the inverse matrix of the weights.
This is one part of the matrix multiplicaton for unmixing the orders.
"""
function dot_mul_last_dim!(order, sim_data, myinv, n, Eps = 1e-7)
    RT = eltype(sim_data)
    CT = Complex{RT}
    contributing = findall(x->abs(x) .> Eps, myinv[n,:])
    sub_matrix = CT.(myinv[n, contributing])
    # mydstidx = ntuple(d->(d==ndims(sim_data)) ? (n:n) : Colon(), ndims(sim_data))
    dv = order # @view orders[mydstidx...] # destination view to write into
    w = sub_matrix[1]
    mymd = ntuple(d->(d==ndims(sim_data)) ? contributing[1] : Colon(), ndims(sim_data))
    sv = @view sim_data[mymd...]
    dv .= w.* (@view sim_data[mymd...])
    for md in 2:length(contributing)
        w = sub_matrix[md]
        mymd = ntuple(d->(d==ndims(sim_data)) ? contributing[md] : Colon(), ndims(sim_data))
        sv = @view sim_data[mymd...]
        dv .+= w.*sv
    end
end

function rfft_size(sz)
    return ntuple((d) -> (d==1) ? sz[d]÷ 2 + 1 : sz[d], length(sz))
end

function get_result_size(sz, upsample_factor)
    return ceil.(Int, sz .* upsample_factor)
end

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
    separate_orders(sim_data, sp)

Separate the orders in the SIM data and apply subpixel shifts. Returns the separated orders and the remaining integer pixelshifts.

Parameters:
+ `sim_data::Array` : simulated SIM data
+ `sp::SIMParams` : SIMParams object

"""
function separate_orders(sim_data, sp)
    myinv = pinv_weight_matrix(sp)
    num_orders = size(sp.peak_phases, 2)
    RT = eltype(sim_data)
    CT = Complex{RT}
    # orders = Array{CT}(undef, size(sim_data)[1:end-1]..., num_orders)
    orders = similar(sim_data, CT, size(sim_data)[1:end-1]..., num_orders)
    pixelsshifts =  Array{NTuple{3, Int}}(undef, num_orders)
    for n=1:num_orders
        contributing = findall(x->x!=0.0, myinv[n,:])
        # myidx = ntuple(d->(d==ndims(sim_data)) ? contributing : Colon(), ndims(sim_data))
        sub_matrix = CT.(myinv[n, contributing])
        sub_matrix = reorient(sub_matrix, Val(ndims(sim_data)))
        # sim_view = @view sim_data[myidx...] 
        mydstidx = ntuple(d->(d==ndims(sim_data)) ? (n:n) : Colon(), ndims(sim_data))
        # sum!(orders[mydstidx...], sim_view .* sub_matrix) 
        for md in 1:size(sub_matrix, ndims(sim_data))
            w = sub_matrix[md]
            mymd = ntuple(d->(d==ndims(sim_data)) ? contributing[md] : Colon(), ndims(sim_data))
            sv = @view sim_data[mymd...]
            if (md==1)
                orders[mydstidx...] .= w.* sv
            else
                orders[mydstidx...] .+= w.* sv
            end
        end
        orders[mydstidx...] .= sum(sim_view .* sub_matrix, dims=ndims(sim_data)) 

        # apply subpixel shifts
        ordershift = .-sp.k_peak_pos[n] .* expand_size(size(sim_data)[1:end-1], ntuple((d)->1, length(sp.k_peak_pos[n]))) ./ 2
        # peakphase = 0.0 # should automatically have been accounted for # .-sp.peak_phases[n, contributing[1]] 
        # peak phases are already accounted for in the weights
        mydstidx = ntuple(d->(d==ndims(sim_data)) ? n : Colon(), ndims(sim_data))
        pixelsshifts[n] = shift_subpixel!((@view orders[mydstidx...]), ordershift, prep, n)
    end
    return orders, pixelsshifts
end

"""
    separate_and_place_orders(sim_data, sp)

Separate the orders in the SIM data and apply subpixel shifts in real space. 
Each separated order is already placed (added) in Fourier space into the result image.

Parameters:
+ `sim_data::Array` : simulated SIM data
+ `sp::SIMParams` : SIMParams object

"""
function separate_and_place_orders(sim_data, sp, prep)
    RT = eltype(sim_data)
    CT = Complex{RT}
    imsz = size(sim_data)[1:end-1]

    otfmul = haskey(prep, :otf) ? prep.otf : one(RT)

    myinv = pinv_weight_matrix(sp)
    num_orders = size(sp.peak_phases, 2)
    order = haskey(prep,:order) ? prep.order : similar(sim_data, CT, imsz...)
    ftorder = haskey(prep,:ftorder) ? prep.ftorder : similar(sim_data, CT, imsz...)
    sz = size(order)
    bsz = ceil.(Int, sz .* prep.upsample_factor) # backwards size
    rec = get_upsampled_rft(sim_data, prep)
    # define the center coordinate of the rft result rec
    bctr = ntuple((d) -> (d==1) ? 1 : bsz[d] .÷ 2 .+ 1, length(bsz))
    # define a shifted center coordinate to account for the flip of even sizes, when appying a flip operation
    bctrbwd = bctr .+ iseven.(bsz)

    # pixelshifts = Array{NTuple{3, Int}}(undef, num_orders)
    # if (otfmul <: AbstractArray)
    #     otfmul = ifftshift(otfmul)
    # end
    for n=1:num_orders
        # apply subpixel shifts
        ordershift = .-sp.k_peak_pos[n] .* expand_size(imsz, ntuple((d)->1, length(sp.k_peak_pos[n]))) ./ 2
        # peakphase = 0.0 # should automatically have been accounted for # .-sp.peak_phases[n, contributing[1]] 
        # peak phases are already accounted for in the weights
        # mydstidx = ntuple(d->(d==ndims(sim_data)) ? n : Colon(), ndims(sim_data))
        ordershift = shift_subpixel!(order, ordershift, prep, n)
        # pixelshifts[n] = ordershift 
        dot_mul_last_dim!(order, sim_data, myinv, n); # writes into order

        # now place (add) the order with possible weights into the result RFFT image
        # myftorder = ft(order)
        # myftorder .*= otfmul
        ifftshift!(ftorder, order)
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

function conj_add!(dst, src)
    dst .+= conj.(src)
end

function add!(dst, src)
    dst .+= src
end

# the function below is not used any more, since it is now part of the separate_and_place_orders function
function place_orders_upsample(orders, pixelshifts, prep)
    rec = get_upsampled_rft(sim_data, prep)

    bctr = ntuple((d) -> (d==1) ? 1 : bsz[d] .÷ 2 .+ 1, length(bsz))
    bctrbwd = bctr .+ iseven.(bsz)  # to account for the flip of even sizes
    for n=1:size(orders, ndims(orders))
        ordershift = pixelshifts[n]

        ids = ntuple(d->(d==ndims(orders)) ? n : Colon(), ndims(orders))
        myftorder = ft(@view orders[ids...])
        myftorder .*= otfmul

        select_region!(myftorder, rec; dst_center = bctr .+ ordershift[1:ndims(rec)], operator! = add!)
        # rec_view = select_region_view(rec, sz; center= bctr .+ ordershift[1:ndims(rec)])
        # rec_view .+= myftorder # writes the FT of the order into the correct region of the final image FT
        idsbwd = ntuple(d-> (size(myftorder, d):-1:1), ndims(myftorder))
        bwd_v = @view myftorder[idsbwd...]
        select_region!(bwd_v, rec; dst_center = bctrbwd .- ordershift[1:ndims(rec)], operator! = conj_add!)
        # rec_view = select_region_view(rec, sz; center= bctrbwd .- ordershift[1:ndims(rec)])
        # idsbwd = ntuple(d-> (size(myftorder, d):-1:1), ndims(myftorder))
        # rec_view .+= conj.(@view myftorder[idsbwd...]) # do the same for the conjugate order at the negative frequency
    end
    return rec, bsz
end

function modify_otf(otf, sigma=0.1, contrast=1.0)
    RT = real(eltype(otf))
    return otf .* (one(RT) .- RT(contrast) .* exp.(-rr2(RT, size(otf), scale=ScaFT)/(2*sigma^2)))
end

function recon_sim_prepare(sim_data, pp::PSFParams, sp::SIMParams, rp::ReconParams, do_preallocate=true)
    sz = size(sim_data)
    imsz = sz[1:end-1]
    h = psf(sz[1:end-1], pp; sampling=sp.sampling)
    # rec = zeros(eltype(sim_data), size(sim_data)[1:end-1])
    RT = eltype(sim_data)
    CT = Complex{RT}
    myotf = modify_otf(fftshift(fft(ifftshift(h))), RT(rp.suppression_sigma), RT(rp.otf_mul))
    ids = ntuple(d->(d==ndims(sim_data)) ? 1 : Colon(), ndims(sim_data))

    ACT = typeof(sim_data[ids...] .+ 0im)
    myotf = ACT(myotf)
    result_rft = get_upsampled_rft(sim_data, (upsample_factor=rp.upsample_factor,))
    myplan_irfft = plan_irfft(result_rft, get_result_size(imsz, rp.upsample_factor)[1]) #, flags=FFTW.MEASURE
    order =  similar(sim_data, CT, imsz...)
    myplan_fft! = plan_fft!(order)  #, flags=FFTW.MEASURE

    prepd = (otf= myotf, upsample_factor=rp.upsample_factor, plan_irfft=myplan_irfft, plan_fft! =myplan_fft!)

    if (do_preallocate)
        result_rft_tmp = get_upsampled_rft(sim_data, prepd)
        result =  similar(sim_data, RT, get_result_size(imsz, rp.upsample_factor)...)
        result_tmp =  similar(sim_data, RT, get_result_size(imsz, rp.upsample_factor)...)
        ftorder =  similar(sim_data, CT, imsz...)
        prepd = (otf= myotf, upsample_factor=rp.upsample_factor, plan_irfft=myplan_irfft, plan_fft! =myplan_fft!,
                result_rft=result_rft, result_rft_tmp=result_rft_tmp, order=order, ftorder=ftorder, result=result, result_tmp=result_tmp)
    end

    dobj = delta(eltype(sim_data), size(sim_data)[1:end-1])
    spd = SIMParams(sp, n_photons = 0.0);
    sim_delta = simulate_sim(dobj, pp, spd);
    ART = typeof(sim_data)
    sim_delta = ART(sim_delta)
    rec_delta = recon_sim(sim_delta, prepd, rp)

    # calculate the final filter
    # rec_otf = ft(rec_delta)
    rec_otf = fftshift(fft(ifftshift(rec_delta)))
    rec_otf ./= maximum(abs.(rec_otf))
    h_goal = RT.(distance_transform(feature_transform(Array(abs.(rec_otf) .< 0.0002))))
    h_goal ./= maximum(abs.(h_goal))

    final_filter = ACT(h_goal) .* conj.(rec_otf)./ (abs2.(rec_otf) .+ RT(rp.wiener_eps))

    final_filter = fftshift(rfft(real.(ifft(ifftshift(final_filter)))), [2,3])

    # the algorithm needs preallocated memory:
    # order: Array{ACT}(undef, size(sim_data)[1:end-1]..., size(sp.peak_phases, 2))
    # result_image: Array{ACT}(undef, size(sim_data)[1:end-1])
    
    if (do_preallocate)
        prep = (otf= myotf, upsample_factor=rp.upsample_factor, final_filter=final_filter, plan_irfft=myplan_irfft, plan_fft! =myplan_fft!,
                result_rft=prepd.result_rft, order=prepd.order, ftorder=prepd.ftorder, result=prepd.result, result_tmp=prepd.result_tmp, result_rft_tmp=prepd.result_rft_tmp)
    else
        prep = (otf= myotf, final_filter=final_filter, upsample_factor=rp.upsample_factor, plan_irfft=myplan_irfft, plan_fft! =myplan_fft!)
    end
    return prep
end

"""
    recon_sim(sim_data, prep, rp::ReconParams)

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
+ `rp::ReconParams` : ReconParams object

"""
function recon_sim(sim_data, prep, rp::ReconParams)

    # first separate (unmix) the orders in real space. This also sets the correct global phase.
    # and apply subpixel shifts 
    # orders, pixelshifts = separate_orders(sim_data, sp)
    # apply FFT
    # and perform Fourier-placement of the orders and upsampling and summation into final ft-image
    # res, bsz = place_orders_upsample(orders, pixelshifts, rp.upsample_factor, prep.otf)

    res, bsz = separate_and_place_orders(sim_data, sp, prep)
    # apply FFT
    # and perform Fourier-placement of the orders and upsampling and summation into final ft-image
    # res, bsz = place_orders_upsample(orders, pixelshifts, rp.upsample_factor, prep.otf)
    
    # apply final frequency-dependent multiplication (filtering)
    if (haskey(prep, :final_filter))
        res .*= prep.final_filter
    end

    # apply IFT of the final ft-image

    res_tmp = haskey(prep, :result_rft_tmp) ? prep.result_rft_tmp : similar(res);
    rec = haskey(prep, :result) ? prep.result : similar(sim_data, eltype(sim_data), bsz...)
    rec_tmp = haskey(prep, :result_tmp) ? prep.result_tmp : similar(rec)

    ifftshift!(res_tmp, res, [2,3])
    if isnothing(prep.plan_irfft)
        rec_tmp .= irfft(res_tmp,  bsz[1])
    else
        mul!(rec_tmp, prep.plan_irfft, res_tmp)
    end
    fftshift!(rec, rec_tmp)
 #   rec = fftshift(irfft(ifftshift(res, [2,3]), bsz[1]))
    # rec = irft(res, bsz[1]) # real.(ift(res))

    # @vt real.(rec) sum(sim_data, dims=3)[:,:,1]
    # @vt ft(real.(rec)) ft(sum(sim_data, dims=3)[:,:,1]) ft(obj)

    return rec
end

function main()

    use_cuda = false;

    lambda = 0.532; NA = 1.4; n = 1.52
    pp = PSFParams(lambda, NA, n);  # 532 nm, NA 0.25 in Water n= 1.33
    sampling = (0.06, 0.06, 0.1)  # 100 nm x 100 nm x 200 nm

    # SIM illumination pattern
    num_phases = 9
    num_directions = 3
    k_peak_pos, peak_phases, peak_strengths = generate_peaks(num_phases, num_directions, 2, 0.48)
    # sp = SIMParams(pp, sampling, 0.0, 0.0, k_peak_pos, peak_phases, peak_strengths)
    sp = SIMParams(pp, sampling, 1000.0, 100.0, k_peak_pos, peak_phases, peak_strengths)

    obj = Float32.(testimage("resolution_test_512"))
    # obj = CuArray(obj)
    # obj .= 1f0
    @time sim_data = simulate_sim(obj, pp, sp);
    if (use_cuda)
        sim_data = CuArray(sim_data);
    end

    # @vv sim_data
    upsample_factor = 1
    rp = ReconParams(0.01, 1.0, upsample_factor)
    prep = recon_sim_prepare(sim_data, pp, sp, rp, true);

    @time rec = recon_sim(sim_data, prep, rp);
    # @profview  rec = recon_sim(sim_data, prep, rp)
    if use_cuda
        # CUDA.@time rec = recon_sim(sim_data, prep, rp);
        @btime CUDA.@sync rec = recon_sim(sim_data, prep, rp);
    else
        @btime rec = recon_sim($sim_data, $prep, $rp);  # 40 ms (512, 10Mb), 55 ms (1024)
    end

    @vt sum(sim_data, dims=3)[:,:,1] rec obj
    @vt ft(real.(rec)) ft(sum(sim_data, dims=3)[:,:,1]) ft(obj)

end
#@vv otf
