# Based on the description in the supplement of Mo et al .. Chen,
# Quantitative structured illumination microscopy via a physical model-based background filtering algorithm reveals actin dynamics
# Nat. Comm. 2023
# https://doi.org/10.1038/s41467-023-38808-8
#
# The notch filter is based on a cut-off Z-coordinate at the Rayleigh limit with a 10x larger distance summing to H_in and H_out
# The notch filter is then (1 - H_out/(H_in + H-_out))

"""
    get_notch(mypsf, pp, sampling, myeps = 1e-6) 

Arguments:
- mypsf: 3D point spread function
- pp: PSF paramters containgin λ, NA and n
- sampling: Th 3rd component will be used to define the cutoff
"""
function psf_notch(mypsf, pp, sampling, rel_eps = 1e-3) 
    rayleight_dist = 2*pp.n*pp.λ/(pp.NA*pp.NA)
    rd = round(Int, rayleight_dist / sampling[3])
    midz = size(mypsf)[3] ÷ 2 + 1
    maxr = min(rd * 10, midz-2)
    psf_if = sum((@view mypsf[:,:,midz-rd:midz+rd]), dims=3)
    psf_oof = (sum((@view mypsf[:,:,midz-maxr:midz-rd-1]), dims=3) + sum((@view mypsf[:,:,midz+rd+1:midz+maxr]), dims=3))[:,:,1]
    ft_if = fftshift(fft(ifftshift(psf_if)))
    ft_oof = fftshift(fft(ifftshift(psf_oof)))
    midftpos = size(ft_if) .÷ 2 .+1
    factor = abs(ft_if[midftpos...])
    RT = real(eltype(mypsf))
    return (1 .- real.(ft_oof .* conj.(ft_if .+ ft_oof) ./ (abs2.(ft_if .+ ft_oof) .+ RT(rel_eps)*(factor*factor))))
end


function gaussian_notch(otf, contrast, sigma) 
    RT = real(eltype(otf))
    real_arr_type(typeof(otf), Val(2))(one(RT) .- RT(contrast) .* exp.(-rr2(RT, size(otf)[1:2], scale=ScaFT)/(2*sigma^2)))
end
