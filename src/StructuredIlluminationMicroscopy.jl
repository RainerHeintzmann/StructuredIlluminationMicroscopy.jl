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

    include("sim_structures.jl")
    include("classical_sim.jl")

end