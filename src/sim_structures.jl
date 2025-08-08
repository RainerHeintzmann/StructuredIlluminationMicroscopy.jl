"""
    SIMParams

a (mutable) structure that holds the parameters for the simulation. See details below.
Constructor:
SIMParams(mypsf, n_photons::Float64, n_photons_bg::Float64, k_peak_pos::Array{NTuple{3, Float64}, 1}, peak_phases::Array{Float64,2}, peak_strengths::Array{Float64,2}, otf_indices::Array{Int,1}=[1], otf_phases::Array{Float64,1}=[0.0])  

Fields:
+ `mypsf` : the point spread function to simulate with.
+ `n_photons::Float64` : the number of photons
+ `n_photons_bg::Float64` : the number of background photons
+ `k_peak_pos::Array{NTuple{3, Float64}, 1}` : peak-positions in k-space, a vector of 3D tuples, in relation to the Nyquist frequency of the image
+ `peak_phases::Array{Float64,2}` : peak-phases in k-space. This is a 2D array with the first dimension being the number of peaks and the second dimension being the number of phases  (i.e. the phases in each image)
+ `peak_strengths::Array{Float64,2}` : peak-intensities in k-space. This is a 2D array with the first dimension being the number of peaks and the second dimension being the number of intensities   (i.e. the intensities of each peak in each image)
+ `otf_indices::Array{Int, 1}` : otf-indices. An array of indices that indicate the OTF to be used for each peak. Note that for three-dimensional OTFs some peaks have associated OTFs where the z-modulation is part of the OTF. Due to refractive index mismatch or misalanement of the optical axis, these OTFs are characterized by a relative phase.
+ `otf_phases::Array{Float64, 1}` : the relative phases of the OTFs, which are approximated as a multiplication of the PSF with a cos(k_z z + phase)
+ `otf_exponent::Float64` : the exponent applied to the magnitude of the OTF, typically 1.0 for unmodified OTFS. 

"""
mutable struct SIMParams
    # psf_params::PSFParams
    mypsf::AbstractArray
    # sampling::NTuple{3, Float64}
    n_photons::Float64
    n_photons_bg::Float64

    k_peak_pos::Array{NTuple{3, Float64}, 1}  # peak-positions in k-space, a vector of 3D tuples, in relation to the Nyquist frequency of the image

    # peak-phases in k-space. This is a 2D array with the first dimension being the number of peaks
    # and the second dimension being the number of phases  (i.e. the phases in each image)
    peak_phases::Array{Float64,2}

    # peak-intensities in k-space. This is a 2D array with the first dimension being the number of peaks
    # and the second dimension being the number of intensities   (i.e. the intensities of each peak in each image)
    # peak_strengths being zero are simply skipped in the calculation
    peak_strengths::Array{Float64,2}

    # otf-indices. An array of indices that indicate the OTF to be used for each peak. Note that for three-dimensional OTFs some peaks have associated OTFs
    # where the z-modulation is part of the OTF. Due to refractive index mismatch or misalanement of the optical axis, these OTFs are characterized by a relative phase.
    otf_indices::Array{Int, 1}

    # the relative phases of the OTFs, which are approximated as a multiplication of the PSF with a cos(k_z z + phase)
    otf_phases::Array{Float64, 1}

    # if not equalt to 1.0, the PSF is modified by applying a power to the corresponding OTF.
    otf_exponent::Float64

    function SIMParams(mypsf, n_photons::Float64, n_photons_bg::Float64, k_peak_pos::Array{NTuple{3, Float64}, 1}, peak_phases::Array{Float64,2}, peak_strengths::Array{Float64,2}, otf_indices::Array{Int,1}=[1], otf_phases::Array{Float64,1}=[0.0], otf_exponent=1.0)  
        if (otf_exponent != 1.0)
                myotf = rfft(mypsf)
                mypsf = irfft(myotf .* abs.(myotf).^otf_exponent ./ (abs.(myotf) .+ 1f-10), size(mypsf,1))
        end
        new(mypsf, n_photons, n_photons_bg, k_peak_pos, peak_phases, peak_strengths, otf_indices, otf_phases, otf_exponent)
    end
    function SIMParams(sp::SIMParams; mypsf=sp.mypsf, n_photons=sp.n_photons, n_photons_bg=sp.n_photons_bg, k_peak_pos=sp.k_peak_pos, peak_phases=sp.peak_phases, peak_strengths=sp.peak_strengths, otf_indices=sp.otf_indices, otf_phases=sp.otf_phases, otf_exponent=sp.otf_exponent)
        new(mypsf, n_photons, n_photons_bg, k_peak_pos, peak_phases, peak_strengths, otf_indices, otf_phases, otf_exponent)
    end
end

"""
    resample_sim_params(sp::SIMParams, resample_factor)

Resamples the SIMParams to a new size by resampling the PSF and adjusting the peak positions.
The `resample_factor` is a number that indicates the factor by which to resample each XY dimension. Z is not resampled.
If the factor is less than or equal to 2, the PSF is resampled, otherwise it is not.

# Arguments:
- `sp::SIMParams`: the SIMParams to resample
- `resample_factor`: a tuple of integers that indicates the factor by which to resample each dimension. If the factor is less than or equal to 2, the PSF is resampled, otherwise it is not.

"""
function resample_sim_params(sp::SIMParams, resample_factor)
    resample_factor = ntuple((d) -> (d<=2) ? resample_factor : 1, length(sp.k_peak_pos[1]))
    # new_sampling =  sp.sampling .* resample_factor 
    new_peakpos = [p .* resample_factor for p in sp.k_peak_pos]
    resampled_psf = let
        if (resample_factor == 1)
            sp.mypsf
        else
            resample_by_rft(sp.mypsf, size(sp.mypsf) .÷ resample_factor[1:ndims(sp.mypsf)])
        end
    end
    return SIMParams(sp; mypsf=resampled_psf, k_peak_pos=new_peakpos)
end

"""
    ReconParams

a (mutable) structure that holds the parameters for the reconstruction algorithm. See details below.
You can use the default constructor `ReconParams()` to get the default values, then overwrite some entries,
or use the named constructor to set the values.

Fields:
+ `notch::Union{AbstractArray{<:Real}, Nothing}` : the notch filter to be applied to the OTFs, can be `nothing` if no notch filter is used.
     It can also be a vector of notch filters, one for each OTF.
+ `suppression_sigma::Float64` : the sigma of the Gaussian suppression filter
+ `suppression_strength::Float64` : the strength of the Gaussian suppression filter
+ `upsample_factor::Int` : the upsampling factor
+ `wiener_eps::Float64` : the epsilon value for the Wiener filter
+ `do_preallocate::Bool` : preallocate memory for the reconstruction
+ `use_measure::Bool` : use the measurement for the reconstruction
+ `double_use::Bool` : use the measurement twice for the reconstruction
+ `preshift_otfs::Bool` : preshift the OTFs
+ `hgoal` : a function defining the goal transfunction. Default is hgoal_one() which yield a constant of one.
+ `hgoal_thresh::Float64` : threshold to determine the hgoal footprint to which the distance transform is applied to
+ `slice_by_slice::Bool` : slice by slice reconstruction
+ `do_deconvolve::Bool` : deconvolve the result instead of Wiener filtering
+ `deconv_lambda::Float64` : the lambda parameter for deconvolution, typically 1.2
+ `keep_hf::Bool` : whether to keep the high-frequency components of the OTFs (default: false)
+ `otf_radius::Float64` : the relative radius of the pupil compared to the sampling limit, used for the OTF masks. zero means that a threshold is used instead of a radius. default is 0.0

"""
mutable struct ReconParams
    notch::Union{AbstractArray{<:Real}, Nothing} # the notch filters to be applied to the OTFs, can be nothing if no notch or gaussian notch filter is used.
    suppression_sigma::Float64 # the sigma of the Gaussian suppression filter, typically 0.2
    suppression_strength::Float64 # the sigma (relative to image size) of the Gaussian suppression filter, typically 0.2
    upsample_factor::Int # the upsampling factor, typically 2 for 2x upsampling
    reference_slice::Int # the slice to use as (alignment) reference for the reconstruction, typically 0 for no reference slice
    wiener_eps::Float64 # epsilon value for the Wiener filter, typically 1e-6
    hgoal_thresh::Float64 # threshold to determine the hgoal footprint to which the distance transform is applied to. Also gets used to determine the OTF footprint for keep_hf below
    do_preallocate::Bool # whether to preallocate memory for the reconstruction
    use_measure::Bool # whether to use the measurement for the reconstruction
    double_use::Bool # whether to reuse some memory in the reconstruction
    preshift_otfs::Bool # whether to preshift the OTFs
    hgoal::Function # whether to use the hgoal algorithm
    slice_by_slice::Bool # whether to reconstruct slice by slice
    do_deconvolve::Bool # whether to deconvolve the result instead of Wiener filtering
    deconv_lambda::Float64 # the lambda parameter for deconvolution, typically 1.2
    keep_hf::Bool # whether to keep the high-frequency components of the OTFs (default: false)
    otf_radius::Float64 # the radius of the pupil, used for the otf masks. zero means that a threshold is used instead of a radius

    function ReconParams(; # constructor with default values
        notch = nothing,
        suppression_sigma = 0.2,
        suppression_strength = 1.0,
        upsample_factor::Int = 2,
        reference_slice::Int = 0,
        wiener_eps = 1e-6,
        hgoal_thresh = 2e-8, # threshold to determine the hgoal footprint to which the distance transform is applied to
        hgoal = hgoal_one, # a function
        do_preallocate=true,
        use_measure=false, # to work with CUDA
        double_use=true,
        preshift_otfs=true,
        slice_by_slice=false, do_deconvolve=false, deconv_lambda=1.2, keep_hf=false)
        otf_radius = 0.0 # 0 means that a threshold is used instead of a radius
        new(notch, Float64(suppression_sigma), 
            Float64(suppression_strength), 
            upsample_factor, 
            reference_slice,
            Float64(wiener_eps),
            Float64(hgoal_thresh),
            do_preallocate, use_measure, double_use, preshift_otfs, hgoal, slice_by_slice, do_deconvolve, deconv_lambda, keep_hf, otf_radius)
    end
end

"""
    PreparationParams(RAT)

a (mutable) structure that holds the parameters for the preparation of the reconstruction algorithm. See details below.
You can use the default constructor `PreparationParams(RAT)` to get the default values, then overwrite some entries,
or use the named constructor to set the values. `RAT` refers to the real array type to be used with the reconstructions it should have
one dimension less than the data to reconstruct but typically the same eltype().

Fields:
+ `pinv_weight_mat::AbstractMatrix` : the pseudo-inverse weight matrix
+ `otfs::Vector{CAT}` : the OTFs
+ `pixelshifts::Vector{NTuple{3, Int}}` : the pixel shifts
+ `slice_by_slice::Bool` : slice by slice reconstruction
+ `upsample_factor::Int` : the upsampling factor
+ `rec_otf`: an OTF describing the forward model of the unmixed and joined data. Can be used for deconvolution
+ `final_filter::CAT` : the final filter
+ `subpixel_shifters::Vector{Any}` : the subpixel shifters
+ `ftorder::CAT` : the Fourier transform order
+ `order::CAT` : the order
+ `result::RAT` : the result
+ `result_rft::CAT` : the result in Fourier space
+ `result_rft_tmp::CAT` : the temporary result in Fourier space
+ `plan_fft!::AbstractFFTs.Plan` : the FFT plan
+ `plan_irfft::AbstractFFTs.Plan` : the iFFT plan

"""    
mutable struct PreparationParams{RAT, CAT} # , CT, D, RT, TA <: AbstractArray{CT, D}, RT = Real{CT}} 
    pinv_weight_mat::AbstractMatrix
    otfs::Vector{CAT}
    pixelshifts::Vector{NTuple{3, Int}}

    slice_by_slice::Bool
    upsample_factor::Int
    final_filter::CAT
    rec_otf::Union{CAT, Nothing}
    subpixel_shifters::Vector{Any}

    ftorder::CAT
    order::CAT
    result::RAT
    result_rft::CAT
    result_rft_tmp::CAT
    plan_fft!::AbstractFFTs.Plan
    plan_irfft::AbstractFFTs.Plan
    deconv_lambda::Float64

    function PreparationParams(RAT::Type)
        CAT = complex_arr_type(RAT, Val(ndims(RAT)))
        pinv_dummy = Array{Float64}(undef, (0,0))
        cat_dummy = CAT(undef, ntuple((d)->1, ndims(CAT)))
        rat_dummy = RAT(undef, ntuple((d)->0, ndims(RAT)))
        plan_dummy = plan_fft!(cat_dummy)
        deconv_lambda=1.2;

        new{RAT, CAT}(pinv_dummy, [cat_dummy,], [(0,0,0),],
                      false, 2, # slice_by_slice, upsample_factor
                      cat_dummy, cat_dummy, # final filter, rec_otf
                      [], # subpixel shifters
                      cat_dummy, cat_dummy, # ftorder, order
                      rat_dummy, cat_dummy, cat_dummy,# result, result_rft, result_rft_tmp
                      plan_dummy, plan_dummy, deconv_lambda)
    end
end

"""
    SIMPattern(p, sp, n, otf_num)

Generate the SIM illumination pattern.
Parameters:
+ `h::PSF` : PSF object, needed only to determine the datatype.
+ sp::SIMParams : SIMParams object
+ `n::Int` : image number
+ `otf_num::Int` : OTF number (only frequencies associated to this OTF number are considered)
                   only those frequencies are considered that contribute to this OTF number
                   as defined by the sp.otf_indices array.

"""
function SIMPattern(h, sp::SIMParams, n, otf_num)
    sim_pattern = zeros(eltype(h), size(h)[1:2])
    pos = idx(eltype(h), size(h)[1:2]) # , offset=CtrFFT)
    for i in eachindex(sp.k_peak_pos)
        k = pi.*sp.k_peak_pos[i][1:2] # sp.k_peak_pos is relative to the Nyquist frequency of the image
        if (otf_num == sp.otf_indices[i] && sp.peak_strengths[n, i] != 0.0)
            strength = sp.peak_strengths[n,i]
            if (strength != 0.0)
                if (norm(k) == 0.0)
                    sim_pattern .+= strength
                else
                    sim_pattern .+= strength .* cos.(dot.(Ref(k), pos) .+ sp.peak_phases[n,i]) 
                end
            end
        end
    end
    return sim_pattern
end

"""
    make_3d_pattern!(sp, num_directions, num_orders)

fills the sp.otf_indices and sp.otf_phases arrays with sensible values for a 3D pattern.
"""
function make_3d_pattern(k_peak_pos, offset_phase=0.0; individual_otfs=false)
    num_peaks = length(k_peak_pos)
    has_kz(p) = (p[3] != 0.0) 

    otf_indices = let 
        if (individual_otfs)
            collect(1:length(k_peak_pos))
        else
            ones(Int, num_peaks)
        end
    end
    if !(individual_otfs)
        otf_indices[has_kz.(k_peak_pos)] .= 2
    end

    otf_phases = zeros(Float64, num_peaks)
    otf_phases[has_kz.(k_peak_pos)] .= offset_phase
    return otf_indices, otf_phases
end
