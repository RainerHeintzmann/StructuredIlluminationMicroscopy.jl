using StructuredIlluminationMicroscopy
using TestImages
using BenchmarkTools
using CUDA
using FourierTools # for resampling and diagnostic purposes
using View5D  # for visualization, @vt etc.
using NDTools  # for select_region

function main()

    use_cuda = false;
    lambda = 0.532; NA = 1.2; n = 1.52
    pp = PSFParams(lambda, NA, n);  # 532 nm, NA 0.25 in Water n= 1.33
    sampling = (0.03, 0.03, 0.1)  # 30 nm x 30 nm x 200 nm

    downsample_factor = 3
    # SIM illumination pattern
    num_directions = 5; num_images =  5*num_directions; num_orders = 3

    rel_peak = 0.80  / (num_orders-1) # peak position relative to sampling limit on fine grid
    k_peak_pos, peak_phases, peak_strengths, otf_indices, otf_phases = generate_peaks(num_images, num_directions, num_orders, rel_peak / (num_orders-1))

    num_photons = 1000.0
    spf = SIMParams(pp, sampling, num_photons, 100.0, k_peak_pos, peak_phases, peak_strengths, otf_indices, otf_phases);

    obj = Float32.(testimage("resolution_test")) # 1920x1920
    obj[(size(obj).÷2 .+1)...] = 2.0 
    obj = select_region(obj, (512*downsample_factor, 512*downsample_factor, 1)) # to yield 512x512 after downsampling

    @time sim_data, sp = simulate_sim(obj, pp, spf, downsample_factor);
    if (use_cuda)
        sim_data = CuArray(sim_data);
    end

    #################################

    # @vv sim_data
    rp = ReconParams() # just use defaults
    rp.upsample_factor = downsample_factor # 1 means no upsampling
    rp.wiener_eps = 1e-4
    rp.suppression_strength = 0.99
    rp.suppression_sigma = 5e-2
    rp.do_preallocate = true
    rp.use_measure=!use_cuda
    rp.double_use=true; rp.preshift_otfs=true; 
    rp.use_hgoal = true
    rp.hgoal_exp = 0.5

    prep = recon_sim_prepare(sim_data, pp, sp, rp); # do preallocate

    @time recon = recon_sim(sim_data, prep, sp);
    wf = (use_cuda) ? sum(sim_data, dims=3)[:,:,1] : resample(sum(sim_data, dims=3)[:,:,1], size(recon))
    # @vt recon
    @vt obj wf recon 

    if use_cuda
        @btime CUDA.@sync recon = recon_sim(sim_data, prep, sp);  
        # 512x512 raw data, upsampling 3x, 7 phase/direction, 5 directions, 7 order total, 9ms
        # 512x512 raw data, upsampling 3x, 5 phase/direction, 5 directions, 5 order total, 6ms
    else
        @btime recon = recon_sim($sim_data, $prep, $sp); 
        # upsample 3, 7 phase/direction, 7 order total, 
        # upsample 3, 5 phase/direction, 5 directions, 5 order total, 39ms
    end

    @vt ft(real.(recon)) ft(wf) ft(obj)
    Base.summarysize(prep) # 9 kB CPU

    sizeof(prep.result)
    sizeof(prep.result_rft)
    sum(sizeof.(values(prep)))/ 1024 /1024 # 42 Mb, or 33 Mb with reuse of memory
    (sum(sizeof.(values(prep))) - sizeof(prep.result))/ 1024 /1024 # 42 Mb, or 33 Mb with reuse of memory
end
#@vv otf
