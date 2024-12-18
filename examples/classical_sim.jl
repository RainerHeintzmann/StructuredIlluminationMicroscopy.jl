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

    num_photons = 0.0 # 1000.00
    spf = SIMParams(pp, sampling, num_photons, 100.0, k_peak_pos, peak_phases, peak_strengths, otf_indices, otf_phases);

    obj = Float32.(testimage("resolution_test_512"));
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
    rp = ReconParams() # just use defaults
    rp.upsample_factor = 2 # 1 means no upsampling
    rp.wiener_eps = 1e-4
    rp.suppression_strength = 0.99
    rp.suppression_sigma = 5e-2
    rp.do_preallocate = true
    rp.use_measure=!use_cuda
    rp.double_use=true; rp.preshift_otfs=true; 
    rp.use_hgoal = true
    rp.hgoal_exp = 0.5

    k_vecs = [(102, 0,0),(-51, 89,0),(-51, -89, 0)]
    k_vecs = nothing
    fff = estimate_parameters(sim_data; pp=sp.psf_params, sampling=sp.sampling, k_vecs=k_vecs,
                            num_directions=num_directions, prefilter_correl=true)

    sp.k_peak_pos
    fff.k_peak_pos
    sp.peak_phases # .- 1.492
    fff.peak_phases # .- 1.492 
    
    prep = recon_sim_prepare(sim_data, pp, sp, rp); # do preallocate
    @time recon = recon_sim(sim_data, prep, sp);
    CUDA.@allowscalar wf = resample(sum(sim_data, dims=3)[:,:,1], size(recon))
    # @vt recon
    @vt obj wf recon 

    prep2 = recon_sim_prepare(sim_data, pp, fff, rp); # do preallocate
    @time recon2 = recon_sim(sim_data, prep2, fff);
    @vt recon2 

    if use_cuda
        @btime CUDA.@sync recon = recon_sim(sim_data, prep, sp);  # 480 µs (one zero order, 256x256)
    else
        @btime recon = recon_sim($sim_data, $prep, $sp);  # 2.2 ms (one zero order, 256x256)
    end

end
#@vv otf
