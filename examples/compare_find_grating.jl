# The aim of this script is to compare various ways to find the grating constants and phases.
# We assume an almost correct (integer) start vector and repeat the search for the grating parameters for a number of times with different photon noise.
using StructuredIlluminationMicroscopy
using TestImages
# using BenchmarkTools
# using CUDA
using FourierTools # for resampling and diagnostic purposes
using View5D  # for visualization, @vt etc
using PointSpreadFunctions # to simulate a realistic PSF
using LinearAlgebra
using Statistics
using Random

function create_object()
    obj = Float32.(testimage("resolution_test_512"));
    obj[(size(obj).÷2 .+1)...] = 2.0 
    if (false)
        obj .= 0.0
        # obj[257,257] = 1.0
        obj[250,250] = 1.0
    end
    return obj
end

function main()
    lambda = 0.532; NA = 1.0; n = 1.52
    pp = PSFParams(lambda, NA, n);  # 532 nm, NA 0.25 in Water n= 1.33
    sampling = (0.06, 0.06, 0.1)  # 100 nm x 100 nm x 200 nm
    obj = create_object()
    mypsf = psf(size(obj), pp, sampling=sampling)

    # SIM illumination pattern
    num_directions = 3; num_images =  3*num_directions; num_orders = 2
    rel_peak = 0.40 # peak position relative to sampling limit on fine grid
    spf = generate_peaks_param(mypsf, num_images, num_directions, num_orders, rel_peak / (num_orders-1))


    # obj[1,1] = 1.0
    # obj = CuArray(obj)
    # obj .= 1f0
    downsample_factor = 2
    @time sim_data, sp = simulate_sim(obj, spf, downsample_factor); # noise-free simulation

    #################################

    # @vv sim_data
    num_photons = 100.00
    num_photons_bg = 100.0 # 100.0 # background photons

    phase_only = false # use phase only for the correlation

    k_vecs = [(102, 0),(-51, 89),(-51, -89)]
    N = 100
    k_err = []
    cor_psf = sp.mypsf # or nothing
    Random.seed!(1234) # for reproducibility
    suppress_sigma = 0.15 # 0.15 # 0.2 # 0.1 # suppress the center of the cross-correlation, if > 0
    otf_exponent = 1.0 # apply this as a power to the OTF, if != 1.0
    k_err = zeros(Float64, N, 3, length(sp.k_peak_pos))
    for n in 1:N
        println("Iteration $n of $N")
        sim_data_n = apply_photon_noise(sim_data, num_photons, num_photons_bg) # apply photon noise to the simulated data, if needed
        ref_dat = sum(sim_data_n, dims=3) # typically widefield image or nothing
        sp_est = estimate_parameters(sim_data_n, cor_psf, ref_dat; 
                                    k_vecs=k_vecs, num_directions=num_directions, ideal_strength=true,
                                    suppress_sigma=suppress_sigma, otf_exponent=otf_exponent,
                                    show_quality=false, verbose=false, subtract_mean=true, phase_only=phase_only)

        for p in 1:length(sp.k_peak_pos)
            k_err[n, :, p] .= sp_est.k_peak_pos[p] .- sp.k_peak_pos[p]
        end
    end
    k_interest = k_err[:,1:2,2:end] # zero order and z position are not interesting
    println("Bias: $(mean(k_interest, dims=1))")
    mean_err = mean(sqrt.(sum(abs2.(k_interest), dims=2)), dims=1)
    println("Mean error: $(mean(mean_err)),  detail: $(mean_err)")
    # ----- No Offset, with subtract_mean=true (default)
    # mypsf, no ref: 0.00057
    # no mypsf, no ref: 0.00043
    # mypsf, wf ref: 0.000573
    # no mypsf, wf ref: 0.000427
    # mypsf, no ref, suppress_sigma=0.2: 0.000277  (best!)
    # mypsf, no ref, suppress_sigma=0.1: 0.000378
    # mypsf, wf ref, suppress_sigma=0.1: 0.000376  
    # mypsf, wf ref, suppress_sigma=0.1, otf_exp 1.5: 0.0048
    # mypsf, wf ref, suppress_sigma=0.1, otf_exp 0.5: 0.00396
    # mypsf, no ref, suppress_sigma=0.1, otf_exp 1.5: 0.00490
    # mypsf, no ref, suppress_sigma=0.1, otf_exp 0.5: 0.00395
    # mypsf, no ref, otf_exp 1.5: 0.00509
    # mypsf, no ref, otf_exp 0.5: 0.0045
    # mypsf, wf ref, suppress_sigma=0.15: 0.000306 (second)

    # mypsf, wf ref, suppress_sigma=0.15, phase_only: 0.000781
    # no psf, no ref, suppress_sigma=0.0, phase_only: 0.00308

    # ------- Offset 100.0, with subtract_mean=true (default):
    # mypsf, no ref: 0.00145
    # no mypsf, no ref: 0.00148
    # mypsf, wf ref: 0.00145
    # no mypsf, wf ref: 0.00147
    # mypsf, no ref, suppress_sigma=0.2: 0.000696
    # mypsf, no ref, suppress_sigma=0.1: 0.000685
    # mypsf, wf ref, suppress_sigma=0.1: 0.000531
    # mypsf, wf ref, suppress_sigma=0.1, otf_exp 1.5: 0.0051
    # mypsf, wf ref, suppress_sigma=0.1, otf_exp 0.5: 0.0042
    # mypsf, no ref, suppress_sigma=0.1, otf_exp 1.5: 
    # mypsf, no ref, suppress_sigma=0.1, otf_exp 0.5: 0.0046
    # mypsf, no ref, otf_exp 1.5: 0.00919
    # mypsf, no ref, otf_exp 0.5: 0.00572
    # mypsf, wf ref, suppress_sigma=0.1, otf_exp: 0.9: 0.0045
    # mypsf, wf ref, suppress_sigma=0.1, otf_exp: 1.1: 0.0047
    # mypsf, wf ref, suppress_sigma=0.05: 0.000565
    # mypsf, wf ref, suppress_sigma=0.15: 0.000506  (best!)

    # mypsf, wf ref, suppress_sigma=0.15, phase_only: 0.00225
    # no psf, no ref, suppress_sigma=0.0, phase_only: 0.00522

end