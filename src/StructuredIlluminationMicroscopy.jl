"""
Structured Illumination Microscopy (SIM) is a super-resolution technique that uses a patterned illumination
to increase the resolution of the microscope. This package provides tools to simulate and reconstruct SIM data.

The package provides the following functionality:
1. Generate SIM illumination patterns
2. Simulate SIM data
3. Reconstruct SIM data

The package is designed to be flexible and can be used with different PSFs and illumination patterns.

# Example
```julia
using StructuredIlluminationMicroscopy
using TestImages

lambda = 0.532; NA = 1.2; n = 1.52
pp = PSFParams(lambda, NA, n);  # 532 nm, NA 0.25 in Water n= 1.33
sampling = (0.06, 0.06, 0.1)  # 100 nm x 100 nm x 200 nm

# define the SIM illumination pattern
num_directions = 5  ; num_images =  5*num_directions; num_orders = 3
k_peak_pos, peak_phases, peak_strengths = generate_peaks(num_images, num_directions, num_orders, 0.48 / (num_orders-1))
num_photons = 0.0
mypsf = psf(pp, sampling=sampling)
sp = SIMParams(mypsf, num_photons, 100.0, k_peak_pos, peak_phases, peak_strengths)

obj = Float32.(testimage("resolution_test_512"))
obj[257, 257] = 2.0
sim_data = simulate_sim(obj, pp, sp);

upsample_factor = 3; wiener_eps = 0.00001; suppression_strength = 0.99
rp = ReconParams(wiener_eps, suppression_strength, upsample_factor)
do_preallocate = true
prep = recon_sim_prepare(sim_data, pp, sp, rp, do_preallocate); # do preallocate

recon = recon_sim(sim_data, prep, sp, rp);
```

# References
1. Heintzmann, R., & Cremer, C. (1999). Laterally modulated excitation microscopy: improvement of resolution by using a diffraction grating. In Proceedings of SPIE - The International Society for Optical Engineering (Vol. 3568, pp. 185–196). https://doi.org/10.1117/12.336826
2. Gustafsson, M. G. L. (2000). Surpassing the lateral resolution limit by a factor of two using structured illumination microscopy. Journal of Microscopy, 198(2), 82–87. https://doi.org/10.1046/j.1365-2818.2000.00710.x
3. Heintzmann, R., Jovin, T. M., & Cremer, C. (2002). Saturated patterned excitation microscopy—a concept for optical resolution improvement. Journal of the Optical Society of America A, 19(8), 1599. https://doi.org/10.1364/JOSAA.19.001599
4. Gustafsson, M. G. L., Shao, L., Carlton, P. M., Wang, C. J. R., Golubovskaya, I. N., Cande, W. Z., … Sedat, J. W. (2008). Three-dimensional resolution doubling in wide-field fluorescence microscopy by structured illumination. Biophysical Journal, 94(12), 4957–4970. https://doi.org/10.1529/biophysj.107.120345
5. Heintzmann, R., & Gustafsson, M. G. L. (2009). Subdiffraction resolution in continuous samples. Nature Photonics, 3(7), 362–364. https://doi.org/10.1038/nphoton.2009.96

# Installation
```julia
using Pkg
Pkg.add("StructuredIlluminationMicroscopy")
```

# Author
- Rainer Heintzmann (
    Friedrich-Schiller-Universität Jena, Institut für Angewandte Physik, Jena, Germany
    Leibniz-Institut für Photonische Technologien e.V., Jena, Germany
    )

# ToDo
- [ ] Add example for two-dimensional excitation patterns
- [ ] Add code for parameter estimation
- [ ] Add support for automatic differentiation
- [ ] Add spatial (Labouesse Masterthesis DOI:10.13140/RG.2.2.25191.66727) reconstruction mode
- [ ] Add more examples
- [ ] Add more tests

"""
module StructuredIlluminationMicroscopy

    using FFTW
    using IndexFunArrays # for rr and delta
    using LinearAlgebra # for dot
    using NDTools  # select_region!
    using Noise # for poisson simulation
    using Images # for distance transform
    using SeparableFunctions
    using Statistics # for mean
    using FindShift # for subpixel correlation

    export PSFParams, SIMParams, ReconParams, PreparationParams
    export generate_peaks, simulate_sim, recon_sim_prepare, recon_sim, make_3d_pattern, get_otfs, get_kz
    export separate_and_place_orders, modify_otf, SIMPattern, get_upsampled_rft, get_result_size, ifftshift_sep!, fftshift_sep!
    export get_shift_subpixel, pinv_weight_matrix, shift_subpixel!, shift_subpixel, shift_subpixel_fft, dot_mul_last_dim!, add!, conj_add!
    export rfft_crop, rfft_size, rfftshift, rifftshift, rfftshift!, rifftshift!, get_rft_center
    export swap_vals!, fftshift!, IntType, force_integer_pixels, estimate_prep_mem, print_mem_usage
    export estimate_parameters
    export preprocess_sim
    export psf_notch, gaussian_notch

    include("preprocess.jl")    
    include("sim_structures.jl")
    include("utils.jl")
    include("simulate_sim.jl")
    include("classical_sim.jl")
    include("parameter_estimation.jl")
    include("notch_filters.jl")

end