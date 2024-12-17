"""
    generate_peaks(num_phases::Int=3, num_directions::Int=3, num_orders=0.0, k0 = 0.9)

generates peaks assuming grating-like illuminations
Parameters:
+ `num_phases::Int` : number of phases
+ `num_directions::Int` : number of directions
+ `num_orders::Int` : number of orders
+ `k0::Float64`: peak frequency of first order (relative to the Nyquist frequency of the image)
"""
function generate_peaks(num_phases::Int=9, num_directions::Int=3, num_orders::Int=2, k1 = 0.9, single_zero_order=true, k1z = 0.0; use_lattice=false, lattice_shift=nothing)
    num_peaks = num_directions * num_orders;
    if (single_zero_order)
        num_peaks -= num_directions - 1 
    end
    k_peak_pos = Array{NTuple{3, Float64}}(undef, num_peaks)
    current_peak = 1
    for d in 0:num_directions-1 # starts at zero
        for o in 0:num_orders-1 # starts at zero
            phi = 2pi*d/num_directions
            k = k1 .* o
            if (o > 0 || d==0 || single_zero_order == false)
                if (k == 0.0) # this is a zero order peak
                    k_peak_pos[current_peak] = (0.0, 0.0, 0.0);  # (d-1)*num_orders + o
                else
                    if (0 < o < num_orders -1 ) # this is a medium order peak, but not the highest one
                        k_peak_pos[current_peak] =  (k .*cos(phi), k .*sin(phi), k1z)
                    else
                        k_peak_pos[current_peak] = k .* (cos(phi), sin(phi), 0.0)
                    end
                end
                current_peak += 1
           end
        end
    end

    if isnothing(lattice_shift) && use_lattice
        # calculate ideal lattic shift here
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
                        if (use_lattice)
                            my_k_vec = k_peak_pos[current_peak]
                            peak_phases[p+1, current_peak] = exp(1îm * dot(lattice_shift, my_k_vec))
                        else
                            peak_phases[p+1, current_peak] = mod(2pi*o*p/phases_per_cycle, 2pi)
                        end
                        peak_strengths[p+1, current_peak] = 1.0
                    end
                    current_peak += 1
                end
            end
        end
    end

    if use_lattice
        peak_strength[:,1] = num_directions
    end    
    otf_indices, otf_phases = make_3d_pattern(k_peak_pos, 0.0); # no offset_phase

    return k_peak_pos, peak_phases, peak_strengths, otf_indices, otf_phases
end

"""
    get_kz(pp, sampling, k1)

estimate the kz positions of the peaks, by assuming that they reside on the perfect excitation OTF outer border

Note that the x-sampling is assumed to be the same as the y-sampling
Parameters:
+ `pp::PSFParams` : PSFParams object
+ `sampling::NTuple` : pixel pitch samplings
+ k1: lateral frequency of the first order peak relativ to Nyquist (assumed to be integerence of zero with first amplitude order) 

"""
function get_kz(pp, sampling::NTuple, k1)
    k0 = 2 .* sampling ./ (pp.λ / pp.n) # radius of the Ewald sphere in the x, y, z directions relative to Nqyuist  
    sin_alpha = pp.NA / pp.n #  k1 / k0[1]
    sin_alpha_prime = sin_alpha - k1 / k0[1]
    if (sin_alpha_prime > 1.0)
        error("The peak frequency is too high for the given sampling, wavelength and refractive index. Reduce the peak frequency or increase the sampling.")
    end
    kz = (cos(asin(sin_alpha_prime)) - cos(asin(sin_alpha))) * k0[3]
    # kz = (1.0 - sqrt(1 - sin_alpha^2)) / k0[3]
    return kz
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
    nd_downsample_factor = ntuple((d) -> (d<=2) ? downsample_factor : 1, length(sz))
    dsz = Int.(round.(sz ./ nd_downsample_factor))

    RT = eltype(obj)

    sim_data = similar(obj, RT, dsz..., size(sp.peak_phases, 1))
    # Generate SIM illumination pattern

    otfs = get_otfs(complex_arr_type(typeof(obj)), sz, pp, sp, true) # unmodified OTFs as rfts for forward simulation
    
    # otf = rfft(ifftshift(h))
    # if (downsample_factor != 1.0)
    #     otf = rfft_crop(otf, dsz) # leads to downsampling
    # end

    for n in 1:size(sp.peak_phases, 1)
        for otf_num = 1:maximum(sp.otf_indices)
            sim_pattern = SIMPattern(h, sp, n, otf_num)
            myidx = ntuple(d->(d==ndims(sim_data)) ? n : Colon(), ndims(sim_data))
            # sim_data[myidx...] .= conv_psf(obj .* sim_pattern, h)
            myrfft = rfft(obj .* sim_pattern)
            # if (downsample_factor != 1.0)
            #     myrfft = rfft_crop(myrfft, dsz) # leads to downsampling
            # end
            down_ids = ntuple(d-> (1:nd_downsample_factor[d]:sz[d]), length(sz))  # this integer downsampling keeps the first pixel and therefore the phases aligned
            if (otf_num == 1)
                sim_data[myidx...] .= irfft(myrfft .* otfs[otf_num], sz[1])[down_ids...]
            else
                sim_data[myidx...] .+= irfft(myrfft .* otfs[otf_num], sz[1])[down_ids...]
            end
        end
    end

    if (sp.n_photons != 0.0)
        sim_data .*= sp.n_photons ./ maximum(sim_data)
        sim_data .= eltype(sim_data).(poisson(Float64.(sim_data))) # cast is due to a bug in the poisson function
    end

    spd = resample_sim_params(sp, downsample_factor)
    return sim_data, spd
end

