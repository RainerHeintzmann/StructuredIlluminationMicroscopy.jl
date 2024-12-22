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

    function SIMParams(mypsf, n_photons::Float64, n_photons_bg::Float64, k_peak_pos::Array{NTuple{3, Float64}, 1}, peak_phases::Array{Float64,2}, peak_strengths::Array{Float64,2}, otf_indices::Array{Int,1}=[1], otf_phases::Array{Float64,1}=[0.0])  
        new(mypsf, n_photons, n_photons_bg, k_peak_pos, peak_phases, peak_strengths, otf_indices, otf_phases)
    end
    function SIMParams(sp::SIMParams; mypsf=sp.mypsf, n_photons=sp.n_photons, n_photons_bg=sp.n_photons_bg, k_peak_pos=sp.k_peak_pos, peak_phases=sp.peak_phases, peak_strengths=sp.peak_strengths, otf_indices=sp.otf_indices, otf_phases=sp.otf_phases)
        new(mypsf, n_photons, n_photons_bg, k_peak_pos, peak_phases, peak_strengths, otf_indices, otf_phases)
    end
end

function resample_sim_params(sp::SIMParams, resample_factor)
    resample_factor = ntuple((d) -> (d<=2) ? resample_factor : 1, length(sp.k_peak_pos[1]))
    # new_sampling =  sp.sampling .* resample_factor 
    new_peakpos = [p .* resample_factor for p in sp.k_peak_pos]
    resampled_psf = let
        if (resample_factor == 1)
            sp.mypsf
        else
            old_size = size(sp.mypsf)
            new_size = Int.(round.(old_size ./ resample_factor[1:ndims(sp.mypsf)]))
            new_size_rft = ntuple((d) -> (d>=2) ? new_size[d] : new_size[d]÷2+1, length(new_size))
            old_rft_center = ntuple((d) -> (d>=2) ? old_size[d]÷2+1 : 1, length(old_size))
            new_rft_center = ntuple((d) -> (d>=2) ? new_size[d]÷2+1 : 1, length(new_size))
            # real.(fftshift(ifft(select_region(fft(ifftshift(sp.mypsf)), new_size))))
            # fftshift(irfft(rifftshift(select_region(rfftshift(rfft(ifftshift(sp.mypsf))), new_size_rft, center=rft_center)), new_size[1]))
            fftshift(irfft(rifftshift(select_region(rfftshift(rfft(ifftshift(sp.mypsf))), new_size_rft, center=old_rft_center, dst_center=new_rft_center)), new_size[1]))
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
+ `suppression_sigma::Float64` : the sigma of the Gaussian suppression filter
+ `suppression_strength::Float64` : the strength of the Gaussian suppression filter
+ `upsample_factor::Int` : the upsampling factor
+ `wiener_eps::Float64` : the epsilon value for the Wiener filter
+ `do_preallocate::Bool` : preallocate memory for the reconstruction
+ `use_measure::Bool` : use the measurement for the reconstruction
+ `double_use::Bool` : use the measurement twice for the reconstruction
+ `preshift_otfs::Bool` : preshift the OTFs
+ `use_hgoal::Bool` : use the hgoal algorithm
+ `hgoal_exp::Float64` : the exponent of the hgoal algorithm
+ `hgoal_thresh::Float64` : threshold to determine the hgoal footprint to which the distance transform is applied to

"""
mutable struct ReconParams
    suppression_sigma::Float64
    suppression_strength::Float64
    upsample_factor::Int
    reference_slice::Int
    wiener_eps::Float64
    hgoal_exp::Float64
    hgoal_thresh::Float64
    do_preallocate::Bool
    use_measure::Bool
    double_use::Bool
    preshift_otfs::Bool
    use_hgoal::Bool
    slice_by_slice::Bool

    function ReconParams(; # constructor with default values
        suppression_sigma = 0.2,
        suppression_strength = 1.0,
        upsample_factor::Int = 2,
        reference_slice::Int = 0,
        wiener_eps = 1e-6,
        hgoal_exp = 0.5, # only used if use_hgoal is true
        hgoal_thresh = 2e-8, # threshold to determine the hgoal footprint to which the distance transform is applied to
        do_preallocate=true,
        use_measure=false, # to work with CUDA
        double_use=true,
        preshift_otfs=true,
        use_hgoal=true, slice_by_slice=false)
        new(Float64(suppression_sigma), 
            Float64(suppression_strength), 
            upsample_factor, 
            reference_slice,
            Float64(wiener_eps),
            Float64(hgoal_exp),
            Float64(hgoal_thresh),
            do_preallocate, use_measure, double_use, preshift_otfs, use_hgoal, slice_by_slice)
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
    subpixel_shifters::Vector{Any}

    ftorder::CAT
    order::CAT
    result::RAT
    result_rft::CAT
    result_rft_tmp::CAT
    plan_fft!::AbstractFFTs.Plan
    plan_irfft::AbstractFFTs.Plan

    function PreparationParams(RAT::Type)
        CAT = complex_arr_type(RAT, Val(ndims(RAT)))
        pinv_dummy = Array{Float64}(undef, (0,0))
        cat_dummy = CAT(undef, ntuple((d)->1, ndims(CAT)))
        rat_dummy = RAT(undef, ntuple((d)->0, ndims(RAT)))
        plan_dummy = plan_fft!(cat_dummy)
        new{RAT, CAT}(pinv_dummy, [cat_dummy,], [(0,0,0),],
                      false, 2,
                      cat_dummy, [],
                      cat_dummy, cat_dummy, rat_dummy, cat_dummy, cat_dummy, plan_dummy, plan_dummy)
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
function make_3d_pattern(k_peak_pos, offset_phase=0.0)
    num_peaks = length(k_peak_pos)
    has_kz(p) = (p[3] != 0.0) 

    otf_indices = ones(Int, num_peaks)
    otf_indices[has_kz.(k_peak_pos)] .= 2
    otf_phases = zeros(Float64, num_peaks)
    otf_phases[has_kz.(k_peak_pos)] .= offset_phase
    return otf_indices, otf_phases
end
