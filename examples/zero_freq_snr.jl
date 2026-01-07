# This script test the SNR of SIM reconstruction to test the zero-frequency paradox
using StructuredIlluminationMicroscopy
using TestImages
using FourierTools # for resampling and diagnostic purposes
using View5D  # for visualization, @vt etc
using PointSpreadFunctions # to simulate a realistic PSF
using Statistics
using Random

function main()
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
    # rel_peak = 0.05 # peak position relative to sampling limit on fine grid
    rel_peak = 0.25 # peak position relative to sampling limit on fine grid
    # k_peak_pos, peak_phases, peak_strengths, otf_indices, otf_phases = generate_peaks(num_images, num_directions, num_orders, rel_peak / (num_orders-1))
    # spf = SIMParams(mypsf, k_peak_pos, peak_phases, peak_strengths, otf_indices, otf_phases);
    spf = generate_peaks_param(mypsf, num_images, num_directions, num_orders, rel_peak / (num_orders-1))

    # obj[1,1] = 1.0
    # obj = CuArray(obj)
    # obj .= 1f0
    downsample_factor = 2
    num_photons = 1000.00
    num_photons_bg = 0.0 # 100.0 # background photons
    @time sim_data, sp = simulate_sim(obj, spf, downsample_factor; n_photons=num_photons, n_photons_bg=num_photons_bg);

    rp = ReconParams() # just use defaults
    rp.upsample_factor = 2 # 1 means no upsampling
    rp.wiener_eps = 4e-3 # 1e-4
    rp.suppression_strength = 0
    rp.suppression_sigma = 5e-2
    rp.do_preallocate = true
    rp.use_measure = true # !use_cuda
    rp.double_use=true; rp.preshift_otfs=true; 
    rp.hgoal = hgoal_j0 # (rrel) -> hgoal_exp(rrel; exponent=0.3)

    use_final_filter = false # if false, the final is not applied, yielding a flat-noise spectrum
    prep = recon_sim_prepare(sim_data, sp, rp; use_final_filter=use_final_filter); # do preallocate
    @time recon = recon_sim(sim_data, prep, sp);

    # CUDA.@allowscalar wf = resample(sum(sim_data, dims=3)[:,:,1], size(recon))
    wf = resample(sum(sim_data, dims=3)[:,:,1], size(recon))
    # @vt obj wf recon 
    # @vt ft(obj) ft(wf) ft(recon) 

    # now use the parameters estimated from the (noisy) data
    # prep2 = recon_sim_prepare(sim_data, sp_est, rp; use_final_filter=use_final_filter); # do preallocate
    # @time recon2 = recon_sim(sim_data, prep2, sp_est);
    # @vt recon recon2 
    # @vt ft(obj) ft(recon) ft(recon2) 

    @time sim_data_p, sp_p = simulate_sim(obj, spf, downsample_factor; n_photons=0, n_photons_bg=0);
    sim_data_p = Float32(num_photons) .* sim_data_p ./ maximum(sim_data_p)

    prep_p = recon_sim_prepare(sim_data_p, sp, rp; use_final_filter=false); # no wiener filtering 
    @time recon_p = recon_sim(sim_data_p, prep_p, sp);

    sim_data = eltype(sim_data_p).(StructuredIlluminationMicroscopy.poisson(Float64.(sim_data_p)))
    prep = recon_sim_prepare(sim_data_p, sp, rp; use_final_filter=false); # no wiener filtering -> constant noise
    # sumimg = 0;
    # sumsqr = 0;
    if (false) # fake the reconstruction OTFs
        # for n=1:length(prep.otfs)
        #     prep.otfs[n] .*= 0;
        #     prep.otfs[n] .+= rr(size(prep.otfs[1])) .< 100;
        # end
        StructuredIlluminationMicroscopy.normalize_otfs!(prep, rp, keep_hf=false)
        for n=2:length(prep.otfs)
            prep.otfs[n] .*= 0;
        end
        # prep.otfs[1] ./= maximum(abs.(prep.otfs[1]))
    end
    wrong_scale = sum(sim_data_p) / sum(recon_p)

    sumraw = [];
    sumsum = [];
    resid_var = 0 .* recon
    N = 100;
    Random.seed!(1234)
    for n = 1:N
        sim_data = eltype(sim_data_p).(StructuredIlluminationMicroscopy.poisson(Float64.(sim_data_p)))
        ss = sum(sim_data);
        push!(sumraw, ss);
        recon = recon_sim(sim_data, prep, sp) * wrong_scale;
        # sumimg = sumimg .+ recon;
        # sumsqr = sumsqr .+ abs2.(recon);
        sr = sum(recon)
        push!(sumsum, sr);
        resid_var .+= abs2.(ft(recon .- recon_p * wrong_scale))
    end
    meanraw = mean(sumraw)
    rawvar = var(sumraw)
    meansum = mean(sumsum)
    recvar = var(sumsum)
    resid_var ./= N
    println("ground truth sum: $(sum(sim_data_p)), raw sum: $(meanraw), recon sum: $(meansum), recon perfect: $(sum(recon_p))")
    println("wrong_scale: $(wrong_scale), raw data variance: $(rawvar), recon sum variance: $(recvar)")
    relnoise = recvar/rawvar
    println("relative noise var: $(relnoise): std.dev.:$(sqrt(relnoise))")
    # @vt recon recon_p recon.-recon_p
    # The last image below should show a flat noise spectrum, if use_final_filter=false
    @vt ft(obj) ft(recon./wrong_scale) ft(recon_p) ft(recon./wrong_scale .- recon_p)
    set_gamma(1.0)
    # Show an averaged sum of the Fourier-space noise:
    @vv resid_var
end
#@vv otf
