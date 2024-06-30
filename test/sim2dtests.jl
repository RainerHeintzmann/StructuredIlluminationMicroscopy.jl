
@testset "generate_peaks" begin

end

@testset "symmetry_2d" begin
    sz = (256, 256)
    obj = zeros(sz)
    mid = sz .÷ 2 .+ 1
    rng = 11
    obj[mid[1]-rng:mid[1]+rng, mid[2]-rng:mid[2]+rng] .= 1.0 # make a symmetric object, causing real transforms

    @test maximum(imag.(StructuredIlluminationMicroscopy.fft(obj))) < 1e-10
    lambda = 0.532; NA = 1.0; n = 1.52
    pp = PSFParams(lambda, NA, n);  # 532 nm, NA 0.25 in Water n= 1.33
    sampling = (0.06, 0.06, 0.1)  # 100 nm x 100 nm x 200 nm

    # SIM illumination pattern
    num_directions = 3; num_images =  3*num_directions; num_orders = 2
    rel_peak = 0.40 # peak position relative to sampling limit on fine grid
    k_peak_pos, peak_phases, peak_strengths, otf_indices, otf_phases = generate_peaks(num_images, num_directions, num_orders, rel_peak / (num_orders-1))
    @test size(k_peak_pos,1) == (num_directions +1)
    @test k_peak_pos[1] == (0.0, 0.0, 0.0)
    @test k_peak_pos[2] == (0.4, 0.0, 0.0)
    @test all(peak_phases[1, :] .== 0.0)
    @test peak_phases[2, 2] ≈ 2pi/3
    @test peak_phases[3, 2] ≈ peak_phases[2, 2] *2
    @test all(peak_strengths[:, 1] .== 1.0)
    @test all(peak_strengths[1:3, 2] .== 1.0)
    @test all(peak_strengths[4:end, 2] .== 0.0)

    # k_peak_pos = force_integer_pixels(k_peak_pos, size(obj))

    num_photons = 0.0
    spf = SIMParams(pp, sampling, num_photons, 100.0, k_peak_pos, peak_phases, peak_strengths, otf_indices, otf_phases)

    downsample_factor = 2
    sim_data, sp = simulate_sim(obj, pp, spf, downsample_factor);
    @test eltype(sim_data) == Float64
    @test size(sim_data) == (sz[1]/2, sz[2]/2, 9)
    @test maximum(imag.(StructuredIlluminationMicroscopy.fft(obj))) < 1e-10
    wf = sum(sim_data, dims=3)[:,:,1]
    @test maximum(imag.(StructuredIlluminationMicroscopy.fft(wf))) < 0.0005 # why so high?

    #################################

    # @vv sim_data
    rp = ReconParams() # just use defaults
    rp.upsample_factor = 2 # 1 means no upsampling
    rp.wiener_eps = 1e-4
    rp.suppression_strength = 0
    rp.suppression_sigma = 5e-2
    rp.do_preallocate = true
    rp.use_measure = true  # !use_cuda
    rp.double_use = true; 
    rp.preshift_otfs = false; 
    rp.use_hgoal = true; rp.hgoal_exp = 0.5

    prep = recon_sim_prepare(sim_data, pp, sp, rp); # do preallocate
    @test all(pointer(prep.otfs[1]) .== pointer(prep.otfs)) # due to the preshift_otfs = false option
    @test sum(abs.(imag.(prep.final_filter))) < 30

    res, bsz = separate_and_place_orders(sim_data, sp, prep)
    @test maximum(abs.(imag.(res))) < 0.001 # why so high?

    @time recon = recon_sim(sim_data, prep, sp);
    @test maximum(imag.(StructuredIlluminationMicroscopy.fft(recon))) < 0.003 # why so high?

    @test size(recon) == size(obj)
    @test sum(recon) ≈ sum(obj)

end
