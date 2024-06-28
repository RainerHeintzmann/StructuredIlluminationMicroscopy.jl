using StructuredIlluminationMicroscopy
using TestImages
using SyntheticObjects
using BenchmarkTools
using CUDA
using FourierTools # for resampling and diagnostic purposes
using View5D  # for visualization, @vt etc.

function main()
    use_cuda = true;

    lambda = 0.532; NA = 1.2; n = 1.33
    pp = PSFParams(lambda, NA, n);  # 532 nm, NA 0.25 in Water n= 1.33
    sampling = (0.05, 0.05, 0.08)  # sampling in the orginal (high-res.) simulation grid. Should be at least twice as good as needed along XY.

    # d_Abbe = lambda / (2*NA) # Abbe resolution limit: 266 nm
    # d_Abbe_rel = sampling[1] / d_Abbe # Abbe resolution limit relative to sampling: 

    # SIM illumination pattern
    num_directions = 3; num_orders = 3; num_images =  (1 + 2*(num_orders-1))*num_directions; 
    rel_peak = 0.1879 # peak position relative to sampling limit on fine grid
    k1z = get_kz(pp, sampling, rel_peak)
    k_peak_pos, peak_phases, peak_strengths, otf_indices, otf_phases = generate_peaks(num_images, num_directions, num_orders, rel_peak / (num_orders-1), true, k1z)
    num_photons = 0.0
    spf = SIMParams(pp, sampling, num_photons, 100.0, k_peak_pos, peak_phases, peak_strengths, otf_indices, otf_phases)

    # obj = filaments3D(Float32, (128, 128, 128));
    obj = filaments3D(Float32, (512, 512, 128));

    downsample_factor = 2 # is only applied in XY, not in Z
    @time sim_data, sp = simulate_sim(obj, pp, spf, downsample_factor);
    if (use_cuda)
        sim_data = CuArray(sim_data);
    end

    #################### process slice by slice

    midz = size(sim_data,3)÷2+1
    sim_slice = sim_data[:,:,midz,:]
    do_preallocate = true; use_measure = !use_cuda
    upsample_factor = 2 # 1 means no upsampling
    wiener_eps = 1e-6
    suppression_strength = 0.99
    suppression_sigma = 1e-3
    rp = ReconParams(suppression_sigma, suppression_strength, upsample_factor, wiener_eps)
    prep = recon_sim_prepare(sim_slice, pp, sp, rp, do_preallocate; use_measure=use_measure); # do preallocate

    # do the slice-by-slice reconstruction
    recon_seq = zeros(Float32, size(prep.result)..., size(sim_data,3))
    for z in axes(sim_data, 3)
        sim_slice = sim_data[:,:,z,:]
        recon_seq[:,:,z] = recon_sim(sim_slice, prep, sp);
    end

    wf = resample(sum(sim_data, dims=ndims(sim_data))[:,:,:,1], size(recon_seq));
    @vt recon_seq wf obj

    ##################### process by 3D SIM algorithm

    # @vv sim_data
    upsample_factor = 2 # 1 means no upsampling
    wiener_eps = 1e-9
    suppression_strength = 0.99
    suppression_sigma = 1e-3
    rp = ReconParams(suppression_sigma, suppression_strength, upsample_factor, wiener_eps)
    do_preallocate = true; use_measure = !use_cuda
    prep = recon_sim_prepare(sim_data, pp, sp, rp, do_preallocate; use_measure=use_measure); # do preallocate

    @time recon = recon_sim(sim_data, prep, sp);
    wf = resample(sum(sim_data, dims=ndims(sim_data))[:,:,:,1], size(recon));
    # @vt recon
    @vt wf recon_seq  recon max.(0, recon) obj 

    @vt ft(recon_seq)  ft(obj) ft(recon) ft(wf)

    if use_cuda
        @btime CUDA.@sync recon = recon_sim(sim_data, prep, sp);  # 100 ms
    else
        @btime recon = recon_sim($sim_data, $prep, $sp);  # 2.036 s (512x512x128)
    end

    # @vt ft(real.(recon)) ft(wf) ft(obj)

end
#@vv otf
