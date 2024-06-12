using View5D
# using FFTW
using FourierTools
using IndexFunArrays # for rr and delta
using PointSpreadFunctions # to calculate PSFs
using TestImages
using LinearAlgebra # for dot
using NDTools
using Noise # for poisson simulation
using Images # for distance transform
using BenchmarkTools

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

    sim_data = Array{eltype(obj)}(undef, size(h)..., size(sp.peak_phases, 1))
    # Generate SIM illumination pattern
    for n in 1:size(sp.peak_phases, 1)
        sim_pattern = SIMPattern(h, sp, n)
        myidx = ntuple(d->(d==ndims(sim_data)) ? n : Colon(), ndims(sim_data))
        sim_data[myidx...] .= conv_psf(obj .* sim_pattern, h)
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
function shift_subpixel!(img, ordershift)
    pixelshift = round.(Int, ordershift)
    subpixelshift = ordershift[1:2] .- pixelshift[1:2]
    if (norm(subpixelshift) == 0.0)
        return pixelshift
    end
    pos = idx(eltype(img), size(img))
    img .*= cis.(dot.(Ref(2pi .* subpixelshift ./ size(img)), pos))
    return pixelshift
end

"""
    dot_mul_last_dim!(orders, sim_data, myinv, n, Eps = 1e-7)

performs the dot product of the last dimension of the SIM data with the inverse matrix of the weights.
This is one part of the matrix multiplicaton for unmixing the orders.
"""
function dot_mul_last_dim!(orders, sim_data, myinv, n, Eps = 1e-7)
    contributing = findall(x->abs(x) .> Eps, myinv[n,:])
    sub_matrix = CT.(myinv[n, contributing])
    mydstidx = ntuple(d->(d==ndims(sim_data)) ? (n:n) : Colon(), ndims(sim_data))
    dv = @view orders[mydstidx...] # destination view to write into
    w = sub_matrix[1]
    mymd = ntuple(d->(d==ndims(sim_data)) ? contributing[1] : Colon(), ndims(sim_data))
    sv = @view sim_data[mymd...]
    dv .= w.* (@view sim_data[mymd...])
    for md in 2:length(contributing)
        w = sub_matrix[md]
        mymd = ntuple(d->(d==ndims(sim_data)) ? md : Colon(), ndims(sim_data))
        sv = @view sim_data[mymd...]
        dv .+= w.*sv
    end
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
    orders = Array{CT}(undef, size(sim_data)[1:end-1]..., num_orders)
    pixelsshifts =  Array{NTuple{3, Int}}(undef, num_orders)
    for n=1:num_orders
        dot_mul_last_dim!(orders, sim_data, myinv, n);
        # apply subpixel shifts
        ordershift = .-sp.k_peak_pos[n] .* expand_size(size(sim_data)[1:end-1], ntuple((d)->1, length(sp.k_peak_pos[n]))) ./ 2
        # peakphase = 0.0 # should automatically have been accounted for # .-sp.peak_phases[n, contributing[1]] 
        # peak phases are already accounted for in the weights
        mydstidx = ntuple(d->(d==ndims(sim_data)) ? n : Colon(), ndims(sim_data))
        pixelsshifts[n] = shift_subpixel!((@view orders[mydstidx...]), ordershift)
    end
    return orders, pixelsshifts
end

function place_orders_upsample(orders, pixelshifts, upsample_factor=2, otfmul=1.0)
    sz = size(orders)[1:end-1]
    bsz = ceil.(Int, sz .* upsample_factor)

    bctr = ntuple((d) -> (d==1) ? 1 : bsz[d] .÷ 2 .+ 1, length(bsz))
    bctrbwd = bctr .+ iseven.(bsz)  # to account for the flip of even sizes
    rec = zeros(eltype(orders), rft_size(bsz)) # since this represents an rft
    for n=1:size(orders, ndims(orders))
        ordershift = pixelshifts[n]

        ids = ntuple(d->(d==ndims(orders)) ? n : Colon(), ndims(orders))
        myftorder = ft(@view orders[ids...])
        myftorder .*= otfmul
        rec_view = select_region_view(rec, sz; center= bctr .+ ordershift[1:ndims(rec)])
        rec_view .+= myftorder # writes the FT of the order into the correct region of the final image FT

        rec_view = select_region_view(rec, sz; center= bctrbwd .- ordershift[1:ndims(rec)])
        idsbwd = ntuple(d-> (size(myftorder, d):-1:1), ndims(myftorder))
        rec_view .+= conj.(@view myftorder[idsbwd...]) # do the same for the conjugate order at the negative frequency
    end
    return rec, bsz
end

function modify_otf(otf, sigma=0.1, contrast=1.0)
    RT = real(eltype(otf))
    return otf .* (one(RT) .- RT(contrast) .* exp.(-rr2(RT, size(otf), scale=ScaFT)/(2*sigma^2)))
end

function recon_sim_prepare(sim_data, pp::PSFParams, sp::SIMParams, rp::ReconParams)
    sz = size(sim_data)
    h = psf(sz[1:end-1], pp; sampling=sp.sampling)
    # rec = zeros(eltype(sim_data), size(sim_data)[1:end-1])
    RT = eltype(sim_data)
    myotf = modify_otf(ft(h), RT(rp.suppression_sigma), RT(rp.otf_mul))
    prep = (otf= myotf,)

    obj = delta(eltype(sim_data), size(sim_data)[1:end-1])
    spd = SIMParams(sp, n_photons = 0.0);
    sim_delta = simulate_sim(obj, pp, spd);
    rec_delta = recon_sim(sim_delta, prep, rp)

    # calculate the final filter
    rec_otf = ft(rec_delta)
    rec_otf ./= maximum(abs.(rec_otf))
    h_goal = RT.(distance_transform(feature_transform(abs.(rec_otf) .< 0.0002)))
    h_goal ./= maximum(abs.(h_goal))

    final_filter = h_goal .* conj.(rec_otf)./ (abs2.(rec_otf) .+ RT(rp.wiener_eps))

    final_filter = rft(real.(ift(final_filter)))
    prep = (otf= myotf, final_filter=final_filter)
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
    orders, pixelshifts = separate_orders(sim_data, sp)
    # apply FFT
    # and perform Fourier-placement of the orders and upsampling and summation into final ft-image
    res, bsz = place_orders_upsample(orders, pixelshifts, rp.upsample_factor, prep.otf)
    
    # apply final frequency-dependent multiplication (filtering)
    if (haskey(prep, :final_filter))
        res .*= prep.final_filter
    end

    # apply IFT of the final ft-image
    rec = irft(res, bsz[1]) # real.(ift(res))

    # @vt real.(rec) sum(sim_data, dims=3)[:,:,1]
    # @vt ft(real.(rec)) ft(sum(sim_data, dims=3)[:,:,1]) ft(obj)

    return rec
end

function main()
    lambda = 0.532; NA = 1.4; n = 1.52
    pp = PSFParams(lambda, NA, n);  # 532 nm, NA 0.25 in Water n= 1.33
    sampling = (0.06, 0.06, 0.1)  # 100 nm x 100 nm x 200 nm

    # SIM illumination pattern
    num_phases = 9
    num_directions = 3
    k_peak_pos, peak_phases, peak_strengths = generate_peaks(num_phases, num_directions, 2, 0.48)
    sp = SIMParams(pp, sampling, 1000.0, 100.0, k_peak_pos, peak_phases, peak_strengths)

    obj = Float32.(testimage("resolution_test_512"))
    # obj .= 1f0
    @time sim_data = simulate_sim(obj, pp, sp);
    @vv sim_data
    upsample_factor = 1
    rp = ReconParams(0.01, 1.0, upsample_factor)
    prep = recon_sim_prepare(sim_data, pp, sp, rp)

    @profview  rec = recon_sim(sim_data, prep, rp)
    @btime rec = recon_sim($sim_data, $prep, $rp);

    @vt sum(sim_data, dims=3)[:,:,1] rec obj
    @vt ft(real.(rec)) ft(sum(sim_data, dims=3)[:,:,1]) ft(obj)

end
#@vv otf
