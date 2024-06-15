module StructuredIlluminationMicroscopy

    using FFTW
    using IndexFunArrays # for rr and delta
    using PointSpreadFunctions # to calculate PSFs
    using LinearAlgebra # for dot
    using NDTools  # select_region!
    using Noise # for poisson simulation
    using Images # for distance transform
    using SeparableFunctions

    export PSFParams, SIMParams, ReconParams
    export generate_peaks, simulate_sim, recon_sim_prepare, recon_sim
    export separate_and_place_orders, modify_otf, SIMPattern, get_upsampled_rft, get_result_size, ifftshift_sep!, fftshift_sep!
    export get_shift_subpixel, pinv_weight_matrix, shift_subpixel!, shift_subpixel, shift_subpixel_fft, dot_mul_last_dim!, add!, conj_add!

    include("sim_structures.jl")
    include("classical_sim.jl")

end