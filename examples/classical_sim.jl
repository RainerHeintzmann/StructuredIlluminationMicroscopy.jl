using View5D
using StructuredIlluminationMicroscopy
using TestImages
using BenchmarkTools
using CUDA

function main()

    use_cuda = true;

    lambda = 0.532; NA = 1.4; n = 1.52
    pp = PSFParams(lambda, NA, n);  # 532 nm, NA 0.25 in Water n= 1.33
    sampling = (0.06, 0.06, 0.1)  # 100 nm x 100 nm x 200 nm

    # SIM illumination pattern
    num_phases = 9 # 7*5  # all phases
    num_directions = 3 # 5  # number of directions
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
    upsample_factor = 2
    rp = ReconParams(0.01, 1.0, upsample_factor)
    prep = recon_sim_prepare(sim_data, pp, sp, rp, true);

    @time rec = recon_sim(sim_data, prep, sp, rp);
    # @profview  rec = recon_sim(sim_data, prep, rp)
    if use_cuda
        # CUDA.@time rec = recon_sim(sim_data, prep, rp);
        @btime CUDA.@sync rec = recon_sim(sim_data, prep, sp, rp);  # 1.9 ms 
    else
        @btime rec = recon_sim($sim_data, $prep, $sp, $rp);  # 22 ms (512x512), 18 ms (one zero order, 512x512), 25 ms (one zero order, 1024x1024)
    end

    @vt sum(sim_data, dims=3)[:,:,1] rec obj
    @vt ft(real.(rec)) ft(sum(sim_data, dims=3)[:,:,1]) ft(obj)

end
#@vv otf
