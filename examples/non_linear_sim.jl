using StructuredIlluminationMicroscopy
using TestImages
using BenchmarkTools
using FourierTools # for resampling and diagnostic purposes
using View5D  # for visualization, @vt etc.
using NDTools  # for select_region
using PointSpreadFunctions

function main()

    lambda = 0.532; NA = 1.2; n = 1.52
    pp = PSFParams(lambda, NA, n);  # 532 nm, NA 0.25 in Water n= 1.33
    sampling = (0.03, 0.03, 0.1)  # 30 nm x 30 nm x 200 nm
    downsample_factor = 3
    obj = Float32.(testimage("resolution_test")) # 1920x1920
    obj[(size(obj).÷2 .+1)...] = 2.0 
    obj = select_region(obj, (512*downsample_factor, 512*downsample_factor, 1)) # to yield 512x512 after downsampling
    mypsf = psf(size(obj), pp, sampling=sampling)

    # SIM illumination pattern
    num_directions = 5; num_phases = 5; num_images =  num_phases*num_directions; num_orders = 3

    rel_peak = 0.80  / (num_orders-1) # peak position relative to sampling limit on fine grid
    spf = generate_peaks_param(mypsf, num_images, num_directions, num_orders, rel_peak / (num_orders-1))

    num_photons = 1000.0

    sat_factor = 1.5 # saturation factor, 0.5 means that the maximum value is 50% of the maximum possible value
    @time sim_data, sp = simulate_sim(obj, spf, downsample_factor; n_photons=num_photons, emission_modification=get_non_linear_saturation(sat_factor));
    # @time sim_data, sp = simulate_sim(obj, spf, downsample_factor; n_photons=num_photons);

    k_vecs = [(154, 0), (47, 146), (124, -90), (124, 90), (-47, 146)] # nothing # [(102, 0),(-51, 89),(-51, -89)]
    # k_vecs = nothing
    sp_est = estimate_parameters(sim_data, sp.mypsf; k_vecs=k_vecs, implied_higher_orders=1,
                            num_directions=num_directions, ideal_strength=true)


    #################################

    # @vv sim_data
    rp = ReconParams() # just use defaults
    rp.upsample_factor = downsample_factor # 1 means no upsampling
    rp.wiener_eps = 1e-4
    rp.suppression_strength = 0.99
    rp.suppression_sigma = 5e-2
    rp.do_preallocate = true
    rp.use_measure= true # !use_cuda
    rp.double_use=true; rp.preshift_otfs=true; 
    rp.hgoal = hgoal_exp 

    use_final_filter = true # use final filter, if true
    prep = recon_sim_prepare(sim_data, sp, rp; use_final_filter=use_final_filter); # do preallocate


    @time recon = recon_sim(sim_data, prep, sp);
    wf = (use_cuda) ? sum(sim_data, dims=3)[:,:,1] : resample(sum(sim_data, dims=3)[:,:,1], size(recon))
    # @vt recon
    @vt obj wf recon 

    recon = recon_sim(sim_data, prep, sp); 

    @vt ft(real.(recon)) ft(wf) ft(obj)
    Base.summarysize(prep) # 9 kB CPU

    sizeof(prep.result)
    sizeof(prep.result_rft)
    sum(sizeof.(values(prep)))/ 1024 /1024 # 42 Mb, or 33 Mb with reuse of memory
    (sum(sizeof.(values(prep))) - sizeof(prep.result))/ 1024 /1024 # 42 Mb, or 33 Mb with reuse of memory
end
#@vv otf
