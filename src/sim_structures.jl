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
    suppression_strength::Float64
    upsample_factor::Int
    wiener_eps::Float64

    function ReconParams(suppression_sigma::Float64 = 0.2, suppression_strength::Float64 = 1.0, upsample_factor::Int = 2, wiener_eps::Float64 = 1e-6)    
        new(suppression_sigma, suppression_strength, upsample_factor, wiener_eps)
    end
end

"""
    SIMPattern(p, sp.peak_pos, sp.peak_phases, sp.peak_strengths)

Generate the SIM illumination pattern.
Parameters:
+ `h::PSF` : PSF object, needed only to determine the datatype.
+ sp::SIMParams : SIMParams object

"""
function SIMPattern(h, sp::SIMParams, n)
    sim_pattern = zeros(eltype(h), size(h))
    pos = idx(eltype(h), size(h)) # , offset=CtrFFT)
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
function generate_peaks(num_phases::Int=9, num_directions::Int=3, num_orders::Int=2, k0 = 0.9, single_zero_order=true)
    num_peaks = num_directions * num_orders;
    if (single_zero_order)
        num_peaks -= num_directions - 1 
    end
    k_peak_pos = Array{NTuple{3, Float64}}(undef, num_peaks)
    current_peak = 1
    for d in 0:num_directions-1 # starts at zero
        for o in 0:num_orders-1 # starts at zero
            theta = 2pi*d/num_directions
            k = k0 .* o
            if (o > 0 || d==0 || single_zero_order == false)
                if (k == 0.0)
                    k_peak_pos[current_peak] = (0.0, 0.0, 0.0);  # (d-1)*num_orders + o
                else
                    k_peak_pos[current_peak] = k .* (cos(theta), sin(theta), 0.0)
                end
                current_peak += 1
           end
        end
    end
    peak_phases = zeros(num_phases, num_peaks)
    peak_strengths = zeros(num_phases, num_peaks)
    phases_per_cycle = num_phases ÷ num_directions
    current_peak = 1
    for p in 0:num_phases-1 # starts at zero
        current_d = p ÷ phases_per_cycle  # directions start at 0
        current_peak = 1
        for d in 0:num_directions-1 # starts at zero
            for o in 0:num_orders-1
                if (o > 0 || d==0 || single_zero_order == false)
                    if (o == 0 || d == current_d)
                        peak_phases[p+1, current_peak] = mod(2pi*o*p/phases_per_cycle, 2pi)
                        peak_strengths[p+1, current_peak] = 1.0
                    end
                    current_peak += 1
                end
            end
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

