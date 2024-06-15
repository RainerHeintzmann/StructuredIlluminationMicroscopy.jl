using View5D
using StructuredIlluminationMicroscopy
using TestImages
using BenchmarkTools
using CUDA

function main()

    use_cuda = false;

    lambda = 0.532; NA = 1.2; n = 1.52
    pp = PSFParams(lambda, NA, n);  # 532 nm, NA 0.25 in Water n= 1.33
    sampling = (0.06, 0.06, 0.1)  # 100 nm x 100 nm x 200 nm

    # SIM illumination pattern
    num_directions = 5  # number of directions, 3
    num_images =  5*num_directions  # all phases 3
    num_orders = 3  # number of orders, 3
    k_peak_pos, peak_phases, peak_strengths = generate_peaks(num_images, num_directions, num_orders, 0.48 / (num_orders-1))
    # sp = SIMParams(pp, sampling, 0.0, 0.0, k_peak_pos, peak_phases, peak_strengths)
    num_photons = 0.0
    sp = SIMParams(pp, sampling, num_photons, 100.0, k_peak_pos, peak_phases, peak_strengths)

    obj = Float32.(testimage("resolution_test_512"))
    obj[257, 257] = 2.0 
    if (false)
        obj .= 0.0
        # obj[257,257] = 1.0
        obj[250,250] = 1.0
    end
    # obj[1,1] = 1.0
    # obj = CuArray(obj)
    # obj .= 1f0
    @time sim_data = simulate_sim(obj, pp, sp);
    if (use_cuda)
        sim_data = CuArray(sim_data);
    end

    # @vv sim_data
    upsample_factor = 3
    wiener_eps = 0.00001
    suppression_strength = 0.99
    rp = ReconParams(wiener_eps, suppression_strength, upsample_factor)
    prep = recon_sim_prepare(sim_data, pp, sp, rp, false);

    @time recon = recon_sim(sim_data, prep, sp, rp);
    @vt recon
    @vt sum(sim_data, dims=3)[:,:,1] recon obj
    # @profview  rec = recon_sim(sim_data, prep, rp)
    if use_cuda
        # CUDA.@time rec = recon_sim(sim_data, prep, rp);
        @btime CUDA.@sync recon = recon_sim(sim_data, prep, sp, rp);  # 1.9 ms (512x512)
        # 512x512 raw data, upsampling 3x, 7 phase/direction, 5 directions, 7 order total, 9ms
        # 512x512 raw data, upsampling 3x, 5 phase/direction, 5 directions, 5 order total, 6ms
    else
        @btime recon = recon_sim($sim_data, $prep, $sp, $rp);  # 22 ms (512x512), 18 ms (one zero order, 512x512), 25 ms (one zero order, 1024x1024)
        # upsample 3, 7 phase/direction, 7 order total, 106 ms
        # upsample 3, 5 phase/direction, 5 directions, 5 order total, 76ms
    end

    @vt ft(real.(recon)) ft(sum(sim_data, dims=3)[:,:,1]) ft(obj)

end
#@vv otf
