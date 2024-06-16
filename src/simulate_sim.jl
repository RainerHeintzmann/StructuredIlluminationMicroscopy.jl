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
+ `downsample_factor::Float64` : downsample factor (e.g. 2.0 means half the number of pixels in each dimension)
"""
function simulate_sim(obj, pp::PSFParams, sp::SIMParams, downsample_factor::Int = 1)
    sz = size(obj)
    h = psf(sz, pp; sampling=sp.sampling);  # sampling is 0.5 x 0.5 x 2.0 µm
    dsz = Int.(round.(sz ./ downsample_factor))

    # sim_data = Array{eltype(obj)}(undef, size(h)..., size(sp.peak_phases, 1))
    sim_data = similar(obj, eltype(obj), dsz..., size(sp.peak_phases, 1))
    # Generate SIM illumination pattern
    otf = rfft(ifftshift(h))
    # if (downsample_factor != 1.0)
    #     otf = rfft_crop(otf, dsz) # leads to downsampling
    # end
    for n in 1:size(sp.peak_phases, 1)
        sim_pattern = SIMPattern(h, sp, n)
        myidx = ntuple(d->(d==ndims(sim_data)) ? n : Colon(), ndims(sim_data))
        # sim_data[myidx...] .= conv_psf(obj .* sim_pattern, h)
        myrfft = rfft(obj .* sim_pattern)
        # if (downsample_factor != 1.0)
        #     myrfft = rfft_crop(myrfft, dsz) # leads to downsampling
        # end
        down_ids = ntuple(d-> (1:downsample_factor:sz[d]), length(sz))  # this integer downsampling keeps the first pixel and therefore the phases aligned
        sim_data[myidx...] .= irfft(myrfft .*otf, sz[1])[down_ids...]
    end

    if (sp.n_photons != 0.0)
        sim_data .*= sp.n_photons ./ maximum(sim_data)
        sim_data .= eltype(sim_data).(poisson(Float64.(sim_data))) # cast is due to a bug in the poisson function
    end

    spd = resample_sim_params(sp, downsample_factor)
    return sim_data, spd
end

