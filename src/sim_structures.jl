mutable struct SIMParams
    psf_params::PSFParams
    sampling::NTuple{3, Float64}
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

    function SIMParams(psf_params::PSFParams, sampling::NTuple{3, Float64}, n_photons::Float64, n_photons_bg::Float64, k_peak_pos::Array{NTuple{3, Float64}, 1}, peak_phases::Array{Float64,2}, peak_strengths::Array{Float64,2})
        new(psf_params, sampling, n_photons, n_photons_bg, k_peak_pos, peak_phases, peak_strengths)
    end
    function SIMParams(sp::SIMParams; sampling=sp.sampling, psf_params=sp.psf_params, n_photons=sp.n_photons, n_photons_bg=sp.n_photons_bg, k_peak_pos=sp.k_peak_pos, peak_phases=sp.peak_phases, peak_strengths=sp.peak_strengths)
        new(psf_params, sampling, n_photons, n_photons_bg, k_peak_pos, peak_phases, peak_strengths)
    end
end

function resample_sim_params(sp::SIMParams, resample_factor)
    new_sampling = sp.sampling .* resample_factor
    new_peakpos = [p .* resample_factor for p in sp.k_peak_pos]
    return SIMParams(sp; sampling=new_sampling, k_peak_pos=new_peakpos)
end

mutable struct ReconParams
    suppression_sigma::Float64
    suppression_strength::Float64
    upsample_factor::Int
    wiener_eps::Float64

    function ReconParams(suppression_sigma::Float64 = 0.2, suppression_strength::Float64 = 1.0, upsample_factor::Int = 2, wiener_eps::Float64 = 1e-6)    
        new(suppression_sigma, suppression_strength, upsample_factor, wiener_eps)
    end
end

"""
    SIMPattern(p, sp.peak_pos, sp.peak_phases, sp.peak_strengths)

Generate the SIM illumination pattern.
Parameters:
+ `h::PSF` : PSF object, needed only to determine the datatype.
+ sp::SIMParams : SIMParams object

"""
function SIMPattern(h, sp::SIMParams, n)
    sim_pattern = zeros(eltype(h), size(h))
    pos = idx(eltype(h), size(h)) # , offset=CtrFFT)
    for i in 1:size(sp.k_peak_pos, 1)
        k = pi.*sp.k_peak_pos[i][1:ndims(h)] # sp.k_peak_pos is relative to the Nyquist frequency of the image
        if (sp.peak_strengths[n,i] != 0.0)
            if (norm(k) == 0.0)
                sim_pattern .+= sp.peak_strengths[n,i]
            else
                sim_pattern .+= cos.(dot.(Ref(k), pos) .+ sp.peak_phases[n,i]) .* sp.peak_strengths[n,i]
            end
        end
    end
    return sim_pattern
end
