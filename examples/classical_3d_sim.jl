using StructuredIlluminationMicroscopy
using TestImages
using SyntheticObjects
using BenchmarkTools
using CUDA
using FourierTools # for resampling and diagnostic purposes
using View5D  # for visualization, @vt etc.
using PointSpreadFunctions
using DeconvOptim

function main()
    use_cuda = false;

    lambda = 0.532; NA = 1.2; n = 1.33
    pp = PSFParams(lambda, NA, n);  # 532 nm, NA 0.25 in Water n= 1.33
    sampling = (0.05, 0.05, 0.08)  # sampling in the orginal (high-res.) simulation grid. Should be at least twice as good as needed along XY.
    obj_sz = (512, 512, 128)
    downsample_factor = 2 # is only applied in XY, not in Z
    detection_sz = (obj_sz[1] ÷ downsample_factor, obj_sz[2] ÷ downsample_factor, obj_sz[3]) # size of the detection grid

    # d_Abbe = lambda / (2*NA) # Abbe resolution limit: 266 nm
    # d_Abbe_rel = sampling[1] / d_Abbe # Abbe resolution limit relative to sampling: 

    # SIM illumination pattern
    num_directions = 3; num_orders = 3; num_images =  (1 + 2*(num_orders-1))*num_directions; 
    rel_peak = 0.1879 # peak position relative to sampling limit on fine grid
    k1z = get_kz(pp, sampling, rel_peak)
    k_peak_pos, peak_phases, peak_strengths, otf_indices, otf_phases = generate_peaks(num_images, num_directions, num_orders, rel_peak / (num_orders-1), true, k1z)
    # obj = filaments3D(Float32, (128, 128, 128));
    obj = filaments3D(Float32, obj_sz);

    num_photons = 100.0
    mypsf = psf(obj_sz, pp; sampling=sampling)
    spf = SIMParams(mypsf, num_photons, 100.0, k_peak_pos, peak_phases, peak_strengths, otf_indices, otf_phases)

    @time sim_data, sp = simulate_sim(obj, spf, downsample_factor);

    if (use_cuda)
        sim_data = CuArray(sim_data);
    end

    # reconstruction parameters for both modes, slice-by-slice and 3D
    rp = ReconParams()
    rp.upsample_factor = 2 # 1 means no upsampling
    rp.wiener_eps = 1e-4
    notch_psf = psf_notch(sp.mypsf, pp, sampling)
    rp.notch = nothing #notch_psf;
    rp.suppression_strength = 0.999 # 0.9 # 0.999
    rp.suppression_sigma = 0.05 # 0.008 # 0.05 # 0.008
    rp.do_preallocate = true; rp.use_measure = !use_cuda
    rp.double_use = true; 
    rp.preshift_otfs= false; # true; 
    rp.use_hgoal = true;
    rp.hgoal_exp = 0.5;

    #################### process slice by slice
    rp.slice_by_slice = true
    prep_seq = prep = recon_seq = nothing; GC.gc(); # to clear the memory    

    # plot(mynotch[:,1,1], title="notch filter")

    prep_seq = recon_sim_prepare(sim_data, sp, rp); # do preallocate
    # do the slice-by-slice reconstruction
    @time recon_seq = recon_sim(sim_data, prep_seq, sp); # 0.8 sec (256x256x128) 

    wf = resample(sum(sim_data, dims=ndims(sim_data))[:,:,:,1], size(recon_seq));
    @vt obj wf recon_seq 

    ##################### process by 3D SIM algorithm

    println("GPU total memory: $(CUDA.total_memory()/1024/1024/1024) Gb")
    CUDA.memory_status(); println()    
    print_mem_usage(sim_data, prep)

    # @vv sim_data
    rp.slice_by_slice = false
    prep_seq = prep = recon = nothing; GC.gc(); # to clear the memory    
    @time prep = recon_sim_prepare(sim_data, sp, rp); # 23 sec

    @time recon = recon_sim(sim_data, prep, sp); # 1.3 sec (256x256x128 raw), 5.2 sec
    wf = resample(sum(sim_data, dims=ndims(sim_data))[:,:,:,1], size(recon));
    # @vt recon
    @vt obj wf recon max.(0, recon) recon_seq  

    @vt  ft(obj) ft(recon_seq)  ft(recon) ft(wf)

    if use_cuda
        @btime CUDA.@sync recon = recon_sim($sim_data, $prep, $sp);  # 100 ms (256x256x128x15), 17 s?? (raw: 512x512x128)
    else
        @btime recon = recon_sim($sim_data, $prep, $sp);  # 1.31 s (raw 256x256x128x15), 2.036 s (512x512x128x15), 5.8 s (raw: 512x512x128x15 new version)
    end

    # @vt ft(real.(recon)) ft(wf) ft(obj)

end
#@vv otf
