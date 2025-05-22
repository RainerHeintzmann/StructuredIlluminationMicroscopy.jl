# This script compares the gaussian notch filter to the notch filter as suggested by Mo et al. 2023.
using StructuredIlluminationMicroscopy
using PointSpreadFunctions
using StructuredIlluminationMicroscopy.FFTW
using Plots

function main()
    lambda = 0.532; NA = 1.2; n = 1.33
    pp = PSFParams(lambda, NA, n);  # 532 nm, NA 0.25 in Water n= 1.33
    sampling = (0.05, 0.05, 0.08)  # sampling in the orginal (high-res.) simulation grid. Should be at least twice as good as needed along XY.
    obj_sz = (512, 512, 256)

    mypsf = psf(obj_sz, pp; sampling=sampling)

    mynotch = psf_notch(mypsf, pp, sampling)
    gauss_notch = gaussian_notch(fft(mypsf), 0.90, 0.008)
    midx, midy = size(mynotch)[1:2] .÷ 2 .+ 1
    psz = 120
    plot(-psz:psz, real.(mynotch[midx-psz:midx+psz,midy]), title="Notch Filter Comparison", label="psf notch filter")
    plot!(-psz:psz,real.(gauss_notch[midx-psz:midx+psz,midy]), label="gaussian notch filter")

end