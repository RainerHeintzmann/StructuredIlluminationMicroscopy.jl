using StructuredIlluminationMicroscopy
using TestImages
# using BenchmarkTools
# using CUDA
using FourierTools # for resampling and diagnostic purposes
using View5D  # for visualization, @vt etc
using PointSpreadFunctions # to simulate a realistic PSF

function main()
    use_cuda = false;

    lambda = 0.532; NA = 1.0; n = 1.52
    pp = PSFParams(lambda, NA, n);  # 532 nm, NA 0.25 in Water n= 1.33
    sampling = (0.06, 0.06, 0.1)  # 100 nm x 100 nm x 200 nm

    obj = Float32.(testimage("resolution_test_512"));
    obj[(size(obj).÷2 .+1)...] = 2.0 
    if (false)
        obj .= 0.0
        # obj[257,257] = 1.0
        obj[250,250] = 1.0
    end
    mypsf = psf(size(obj), pp, sampling=sampling)

    # SIM illumination pattern
    num_directions = 3; num_phases = 3; num_images =  num_phases*num_directions; num_orders = 2
    rel_peak = 0.40 # peak position relative to sampling limit on fine grid
    # k_peak_pos, peak_phases, peak_strengths, otf_indices, otf_phases = generate_peaks(num_images, num_directions, num_orders, rel_peak / (num_orders-1))
    # spf = SIMParams(mypsf, k_peak_pos, peak_phases, peak_strengths, otf_indices, otf_phases);
    spf = generate_peaks_param(mypsf, num_images, num_directions, num_orders, rel_peak / (num_orders-1))


    # obj[1,1] = 1.0
    # obj = CuArray(obj)
    # obj .= 1f0
    downsample_factor = 2
    num_photons = 100.00
    num_photons_bg = 0.0 # 100.0 # background photons
    @time sim_data, sp = simulate_sim(obj, spf, downsample_factor; n_photons=num_photons, n_photons_bg=num_photons_bg);
    if (use_cuda)
        sim_data = CuArray(sim_data);
    end

    #################################

    # @vv sim_data

    k_vecs = [(102, 0),(-51, 89),(-51, -89)]
    # k_vecs = nothing
    sp_est = estimate_parameters(sim_data, sp.mypsf; k_vecs=k_vecs,
                            num_directions=num_directions, ideal_strength=true)

    @show sp.k_peak_pos
    @show sp_est.k_peak_pos
    @show sp.peak_phases # .- 1.492
    @show sp_est.peak_phases # .- 1.492 
    @show sp.peak_strengths
    @show sp_est.peak_strengths

    rp = ReconParams() # just use defaults
    rp.upsample_factor = 2 # 1 means no upsampling
    rp.wiener_eps = 4e-3 # 1e-4
    rp.suppression_strength = 0.99
    rp.suppression_sigma = 5e-2
    rp.do_preallocate = true
    rp.use_measure = !use_cuda
    rp.double_use=true; rp.preshift_otfs=true; 
    rp.hgoal = hgoal_j0 # (rrel) -> hgoal_exp(rrel; exponent=0.3)

    use_final_filter = true # if false, the final is not applied, yielding a flat-noise spectrum
    prep = recon_sim_prepare(sim_data, sp, rp; use_final_filter=use_final_filter); # do preallocate
    @time recon = recon_sim(sim_data, prep, sp);

    # CUDA.@allowscalar wf = resample(sum(sim_data, dims=3)[:,:,1], size(recon))
    wf = resample(sum(sim_data, dims=3)[:,:,1], size(recon))
    # @vt recon
    @vt obj wf recon 
    @vt ft(obj) ft(wf) ft(recon) 

    # now use the parameters estimated from the (noisy) data
    prep2 = recon_sim_prepare(sim_data, sp_est, rp; use_final_filter=use_final_filter); # do preallocate
    @time recon2 = recon_sim(sim_data, prep2, sp_est);
    @vt recon recon2 
    @vt ft(obj) ft(recon) ft(recon2) 

    if (false) # compare with perfect data to see the noise
        @time sim_data_p, sp_p = simulate_sim(obj, spf, downsample_factor; n_photons=0, n_photons_bg=0);
        sim_data_p = num_photons .* sim_data_p ./ maximum(sim_data_p)

        prep_p = recon_sim_prepare(sim_data_p, sp, rp; use_final_filter=false); # no wiener filtering 
        @time recon_p = recon_sim(sim_data_p, prep_p, sp);

        prep = recon_sim_prepare(sim_data, sp, rp; use_final_filter=false); # no wiener filtering -> constant noise
        @time recon = recon_sim(sim_data, prep, sp);

        # @vt recon recon_p recon.-recon_p
        # The last image below should show a flat noise spectrum, if use_final_filter=false
        @vt ft(obj) ft(recon) ft(recon_p) ft(recon.-recon_p)
    end

    if use_cuda
        @btime CUDA.@sync recon = recon_sim(sim_data, prep, sp);  # 480 µs (one zero order, 256x256)
    else
        @btime recon = recon_sim($sim_data, $prep, $sp);  # 2.2 ms (one zero order, 256x256)
    end

end
#@vv otf
