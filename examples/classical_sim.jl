using StructuredIlluminationMicroscopy
using TestImages
using BenchmarkTools
using CUDA
using FourierTools # for resampling and diagnostic purposes
using View5D  # for visualization, @vt etc.

function main()
    use_cuda = false;

    lambda = 0.532; NA = 1.0; n = 1.52
    pp = PSFParams(lambda, NA, n);  # 532 nm, NA 0.25 in Water n= 1.33
    sampling = (0.06, 0.06, 0.1)  # 100 nm x 100 nm x 200 nm

    # SIM illumination pattern
    num_directions = 3; num_images =  3*num_directions; num_orders = 2
    rel_peak = 0.40 # peak position relative to sampling limit on fine grid
    k_peak_pos, peak_phases, peak_strengths, otf_indices, otf_phases = generate_peaks(num_images, num_directions, num_orders, rel_peak / (num_orders-1))

    num_photons = 0.0
    spf = SIMParams(pp, sampling, num_photons, 100.0, k_peak_pos, peak_phases, peak_strengths, otf_indices, otf_phases)

    obj = Float32.(testimage("resolution_test_512"))
    obj[(size(obj).÷2 .+1)...] = 2.0 
    if (false)
        obj .= 0.0
        # obj[257,257] = 1.0
        obj[250,250] = 1.0
    end
    # obj[1,1] = 1.0
    # obj = CuArray(obj)
    # obj .= 1f0
    downsample_factor = 2
    @time sim_data, sp = simulate_sim(obj, pp, spf, downsample_factor);
    if (use_cuda)
        sim_data = CuArray(sim_data);
    end

    #################################

    # @vv sim_data
    upsample_factor = 2 # 1 means no upsampling
    wiener_eps = 0.00001
    suppression_strength = 0.99
    suppression_sigma = 1e-3
    rp = ReconParams(suppression_sigma, suppression_strength, upsample_factor, wiener_eps)
    do_preallocate = true; use_measure = !use_cuda
    prep = recon_sim_prepare(sim_data, pp, sp, rp, do_preallocate; use_measure=use_measure); # do preallocate

    @time recon = recon_sim(sim_data, prep, sp);
    wf = resample(sum(sim_data, dims=3)[:,:,1], size(recon))
    # @vt recon
    @vt wf recon obj
    if use_cuda
        @btime CUDA.@sync recon = recon_sim(sim_data, prep, sp);  # 1.9 ms (512x512)
    else
        @btime recon = recon_sim($sim_data, $prep, $sp);  # 22 ms (512x512), 18 ms (one zero order, 512x512), 25 ms (one zero order, 1024x1024)
    end

    # @vt ft(real.(recon)) ft(wf) ft(obj)

end
#@vv otf
