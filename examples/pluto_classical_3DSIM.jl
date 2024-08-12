### A Pluto.jl notebook ###
# v0.19.45

using Markdown
using InteractiveUtils

# ╔═╡ bdda21e0-564b-11ef-1535-310e19b5b039
using Pkg

# ╔═╡ fef37db9-6f88-4482-826c-fece0f1ea8de
Pkg.activate(".")

# ╔═╡ 88e7a870-e7d6-4186-9ef2-a3b544b50a2e
using TestImages, PointSpreadFunctions, TestImages, StructuredIlluminationMicroscopy, ImageShow, PlutoUI, FourierTools, NDTools, View5D, SyntheticObjects, BenchmarkTools #packages

# ╔═╡ 08281294-23f1-4d4f-82dc-2a95edf79f70
begin
	    lambda = 0.532; NA = 1.2; n = 1.33
    pp = PSFParams(lambda, NA, n);  # 532 nm, NA 0.25 in Water n= 1.33
    sampling = (0.05, 0.05, 0.08)  # sampling in the orginal (high-res.) simulation grid. Should be at least twice as good as needed along XY.
    obj_sz = (512, 512, 128)

    # d_Abbe = lambda / (2*NA) # Abbe resolution limit: 266 nm
    # d_Abbe_rel = sampling[1] / d_Abbe # Abbe resolution limit relative to 
end

# ╔═╡ 36da6888-523b-4cd1-ad23-9498603b640a
begin
	# SIM illumination pattern
    num_directions = 3; num_orders = 3; num_images =  (1 + 2*(num_orders-1))*num_directions; 
    rel_peak = 0.1879 # peak position relative to sampling limit on fine grid
    k1z = get_kz(pp, sampling, rel_peak)
    k_peak_pos, peak_phases, peak_strengths, otf_indices, otf_phases = generate_peaks(num_images, num_directions, num_orders, rel_peak / (num_orders-1), true, k1z)
    num_photons = 0.0
    spf = SIMParams(pp, sampling, num_photons, 100.0, k_peak_pos, peak_phases, peak_strengths, otf_indices, otf_phases)
end

# ╔═╡ 6d51b0d5-eeda-4085-939e-3950e92dc527
begin 
	#obj = filaments3D(Float32, (128, 128, 128));
    obj = filaments3D(Float32, obj_sz);

    downsample_factor = 2 # is only applied in XY, not in Z
    @time sim_data, sp = simulate_sim(obj, pp, spf, downsample_factor);

end

# ╔═╡ 9f1658aa-6b1f-49b2-b140-22356af6b0f1
begin
	# reconstruction parameters for both modes, slice-by-slice and 3D
    rp = ReconParams()
    rp.upsample_factor = 2 # 1 means no upsampling
    rp.wiener_eps = 1e-9
    rp.suppression_strength = 0.99
    rp.suppression_sigma = 1e-3
    rp.do_preallocate = true; rp.use_measure = true
    rp.double_use = true; 
    rp.preshift_otfs= false; # true; 
    rp.use_hgoal = true;
    rp.hgoal_exp = 0.5;
end

# ╔═╡ fc5f45c3-e252-475e-a1db-8aaa34e59007
begin
	#################### process slice by slice
    rp.slice_by_slice = true
    prep_seq = prep = recon_seq = nothing; GC.gc(); # to clear the memory    
    prep_seq = recon_sim_prepare(sim_data, pp, sp, rp); # do preallocate
    # do the slice-by-slice reconstruction
    @time recon_seq = recon_sim(sim_data, prep_seq, sp); # 0.8 sec (256x256x128) 

    wf = resample(sum(sim_data, dims=ndims(sim_data))[:,:,:,1], size(recon_seq));
    @vt obj wf recon_seq 
end

# ╔═╡ 01f6623a-18b4-4b6a-9bc6-3512cc7288e6
begin
	#println("GPU total memory: $(CUDA.total_memory()/1024/1024/1024) Gb")
	   #CUDA.memory_status(); println()    
	    print_mem_usage(sim_data, prep)
end

# ╔═╡ 0d9875d9-0376-4466-b788-c0eab2156832
imageshow(wf)

# ╔═╡ Cell order:
# ╠═bdda21e0-564b-11ef-1535-310e19b5b039
# ╠═fef37db9-6f88-4482-826c-fece0f1ea8de
# ╠═88e7a870-e7d6-4186-9ef2-a3b544b50a2e
# ╠═08281294-23f1-4d4f-82dc-2a95edf79f70
# ╠═36da6888-523b-4cd1-ad23-9498603b640a
# ╠═6d51b0d5-eeda-4085-939e-3950e92dc527
# ╠═9f1658aa-6b1f-49b2-b140-22356af6b0f1
# ╠═fc5f45c3-e252-475e-a1db-8aaa34e59007
# ╠═01f6623a-18b4-4b6a-9bc6-3512cc7288e6
# ╠═0d9875d9-0376-4466-b788-c0eab2156832
