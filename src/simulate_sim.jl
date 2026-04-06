"""
    generate_peaks(num_images::Int=3, num_directions::Int=3, num_orders=0.0, k0 = 0.9)

generates peaks assuming grating-like illuminations
Parameters:
+ `num_images::Int` : number of phases
+ `num_directions::Int` : number of directions
+ `num_orders::Int` : number of orders
+ `k0::Float64`: peak frequency of first order (relative to the Nyquist frequency of the image)
+ individual_otfs: if true, each k-shift gets its own otf index assigned in the peak generation (default: false)
"""
function generate_peaks(num_images::Int=9, num_directions::Int=3, num_orders::Int=2, k1 = 0.9, single_zero_order=true, k1z = 0.0; use_lattice=false, lattice_shift=nothing, individual_otfs=false)
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
    peak_phases = zeros(num_images, num_peaks)
    peak_strengths = zeros(num_images, num_peaks)
    phases_per_cycle = num_images ÷ num_directions
    current_peak = 1
    for p in 0:num_images-1 # starts at zero
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
    otf_indices, otf_phases = make_3d_pattern(k_peak_pos, 0.0; individual_otfs=individual_otfs); # no offset_phase

    return k_peak_pos, peak_phases, peak_strengths, otf_indices, otf_phases
end

"""
    generate_peaks_param(mypsf, vargs...; kwargs...)

calls `generate_peaks` and returns a SIMParams object with the generated peak information packaged into the SIMParams structure.

Parameters:
+ `mypsf::Array` : PSF to use for the SIMParams
+ `vargs...` : arguments to pass to `generate_peaks`
+ `kwargs...` : keyword arguments to pass to `generate_peaks`
"""
function generate_peaks_param(mypsf, vargs...; kwargs...)
    k_peak_pos, peak_phases, peak_strengths, otf_indices, otf_phases =generate_peaks(vargs...; kwargs...)
    return SIMParams(mypsf, k_peak_pos, peak_phases, peak_strengths, otf_indices, otf_phases);
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
    randomize_phases(spf::SIMParams, ΔΦ = 0.1)

applies a random phase shift uniformly distributed between -ΔΦ and ΔΦ to the phase data
in a copy of spf which is returned.
"""
function randomize_phases(spf::SIMParams, ΔΦ = 0.1)
    modified_spf = SIMParams(spf)
    for order in 2:size(spf.peak_strengths,2)
        for img_num in 1:size(spf.peak_strengths,1)
            if (spf.peak_strengths[img_num, order] != 0)
                modified_spf.peak_phases[img_num, order] += (2*rand()-1)*ΔΦ
            end
        end
    end
    return modified_spf
end


"""
    simulate_sim(obj, sp::SIMParams, downsample_factor::Int = 1; n_photons = 0.0, n_photons_bg=0.0, emission_modification=identity)

Simulate SIM data.
Parameters:
+ `obj::Array` : object or a generator function that generates an object of the same size as the PSF in `sp.mypsf` via calling `obj(size(sp.mypsf))`.
+ `sp::SIMParams` : SIMParams object
+ `downsample_factor::Float64` : downsample factor (e.g. 2.0 means half the number of pixels in each dimension)
+ `n_photons::Float64` : number of photons to simulate (default: 0.0, no photons simulated)
+ `n_photons_bg::Float64` : background photons to simulate (default: 0.0, no background photons simulated)
+ `emission_modification::Function` : (non-linear) function to apply to the object before simulating the emission (default: identity function, no modification)

returned are the simulated data and the SIMParams object with the downsampled PSF and peak phases.
# Example:
see the example folder for a complete example.
"""
function simulate_sim(obj, sp::SIMParams, downsample_factor::Int = 1; n_photons = 0.0, n_photons_bg=0.0, emission_modification=identity)
    if (isa(obj, Function))
        obj = eltype(spf.mypsf).(obj_generator(sp.mypsf)) # generate an object
    end

    sz = size(obj)
    h = sp.mypsf
    nd_downsample_factor = ntuple((d) -> (d<=2) ? downsample_factor : 1, length(sz))
    dsz = Int.(round.(sz ./ nd_downsample_factor))

    RT = eltype(obj)

    sim_data = similar(obj, RT, dsz..., size(sp.peak_phases, 1))
    # Generate SIM illumination pattern

    otfs = get_otfs(complex_arr_type(typeof(obj)), sz, sp, true) # unmodified OTFs as rfts for forward simulation
    
    # otf = rfft(ifftshift(h))
    # if (downsample_factor != 1.0)
    #     otf = rfft_crop(otf, dsz) # leads to downsampling
    # end

    for n in 1:size(sp.peak_phases, 1)
        for otf_num = 1:maximum(sp.otf_indices)
            sim_pattern = SIMPattern(h, sp, n, otf_num)
            myidx = ntuple(d->(d==ndims(sim_data)) ? n : Colon(), ndims(sim_data))
            # sim_data[myidx...] .= conv_psf(obj .* sim_pattern, h)
            normal_emission = obj .* sim_pattern
            modified_emission = emission_modification.(normal_emission)
            if (n == 1 && emission_modification != identity)
                ratio = maximum(normal_emission) / maximum(modified_emission)
                if (ratio < 1.2 && ratio > 0.8)
                    @warn "emission modification was only $(ratio)"
                end
                println("emission modification was $(ratio)")
            end
            myrfft = rfft(modified_emission)
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

    sim_data = apply_photon_noise(sim_data, n_photons, n_photons_bg) # apply photon noise to the simulated data, if needed

    spd = resample_sim_params(sp, downsample_factor) # also upsamples the psf
    return sim_data, spd
end

"""
    apply_photon_noise(sim_data, n_photons, n_photons_bg=0.0)

Apply photon noise to the simulated data.
Parameters:
+ `sim_data::Array` : simulated data to apply photon noise to
+ `n_photons::Float64` : number of photons to simulate
+ `n_photons_bg::Float64` : background photons to simulate (default: 0.0, no background photons simulated)
"""
function apply_photon_noise(sim_data, n_photons, n_photons_bg=0.0)
    if (n_photons != 0.0)
        sim_data_n = sim_data .* eltype(sim_data)(n_photons) ./ maximum(sim_data)
        sim_data_n .+= eltype(sim_data)(n_photons_bg);
        sim_data_n .= eltype(sim_data).(poisson(Float64.(sim_data_n))) # cast is due to a bug in the poisson function
    else
        sim_data_n = sim_data
    end
    return sim_data_n
end


"""
    simulate_sim_3d(spf, mypsf; n_photons = 1000, n_photons_bg=1.2f0, downsample_factor = 2, emission_modification=identity)
Simulate a 3D SIM data set with a given 3D-PSF (mypsf) and (potentially two-dimensional) SIM  parameters (spf) as used for later object reconstruction.

Parameters:
+ `spf::SIMParams` : SIM parameters (2D)
+ `mypsf::Array` : 3D PSF
+ `obj` : the object to use or a function that generates a 3D object of size (bsz) when called with bsz as argument, will be cast into the eltype that the 2D psf in spf has.
+ `n_photons::Int` : number of photons to simulate
+ `n_photons_bg::Float32` : background photons to simulate
+ `downsample_factor::Int` : downsample factor, which upsamples to generate the object (e.g. 2 means half the number of pixels in each dimension)

returned are the simulated data and the object used for simulation.
"""
function simulate_sim_3d(obj, spf, mypsf; n_photons = 1000, n_photons_bg=1.2f0, downsample_factor = 2, emission_modification=identity)
    bsz = (size(mypsf,1)*downsample_factor, size(mypsf,2)*downsample_factor, size(mypsf,3))
    if (isa(obj, Function))
        obj = eltype(spf.mypsf).(obj(bsz)) # generate a 3D object
    else 
        if (size(obj) != bsz)
            error("The object size $(size(obj)) does not match enlarged 3D PSF size $(bsz). Please provide an object of the correct size or a function that generates such an object.")
        end
        obj = eltype(spf.mypsf).(obj) # generate a 3D object
    end

    tmppsf = spf.mypsf
    spf.mypsf = mypsf
    # spd = StructuredIlluminationMicroscopy.resample_sim_params(spf, downsample_factor); # resample the SIMParams to the new size

    # spf.mypsf = resample(mypsf, (downsample_factor, downsample_factor, 1).*size(mypsf)) # set the PSF in the SIMParams
    spf.mypsf = resample_by_rft(mypsf, (downsample_factor, downsample_factor, 1).*size(mypsf)) # set the PSF in the SIMParams
    tmp_k_est = spf.k_peak_pos
    df = (downsample_factor, downsample_factor, 1)
    spf.k_peak_pos = [k_peak_pos./df  for k_peak_pos in (spf.k_peak_pos)]    
    measured, sp = simulate_sim(obj, spf, downsample_factor; n_photons = n_photons, n_photons_bg = n_photons_bg, emission_modification=emission_modification);
    spf.k_peak_pos = tmp_k_est
    spf.mypsf = tmppsf # restore the PSF in the SIMParams

    return measured, obj
end

"""
get_non_linear_saturation(sat_intensity)

Apply non-linear hyperbolic saturation to the simulated data.
returns a function:
res = intensity / (1.0 + intensity / sat_intensity)
Parameters:
+ `intensity::Array`: intensity to apply non-linear saturation to
+ `sat_intensity::Float64` : saturation intensity, 0.5 means that the maximum value is 50% of the maximum possible value
"""
function get_non_linear_saturation(sat_intensity)
    return (intensity) -> intensity / (1.0 + intensity / sat_intensity)
end