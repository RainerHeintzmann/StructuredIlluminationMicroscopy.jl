# This file contains a number of possible hgoal functions for the
# StructuredIlluminationMicroscopy package.
# they are each called with a size argument and sometimes have more parameters

export hgoal_gabor_cos, hgoal_j0, hgoal_leaver_1D, hgoal_leaver_2D,
       hgoal_lucosz_sufficient, hgoal_standard_otf, hgoal_hanning,
       hgoal_sonine, hgoal_exp, hgoal_one

# helper functions to evaluate them
function strehl_ratio(mypsf)
    return maximum(mypsf) ./ sum(mypsf)
end

function second_moment(mypsf)
    return sum(rr2(size(mypsf)) .* abs.(mypsf)) / sum(abs.(mypsf))
end

const first_bessel_crossing = 2.404825557695773

"""
    hgoal_gabor_cos(rrel)
Minimizes the variance of the abs2 of the Fourier transform in 1D
This is described in the Gabor 1945 paper "Theory of Communications", 429-441.
See equation 1.36, but written with the central frequency being zero. This minimizes the variance of the abs2 of the Fourier transform.
Without the square you get higher resolution but negative ringing. This is how it is used here.
"""
function hgoal_gabor_cos(rrel)
    rrel = min(one(typeof(rrel)), rrel)  # ensure rrel is at least one
    return cos(pi*rrel/2)  # 1D
end

"""
    hgoal_j0(rrel)
Minimizes the variance of the abs2 of the Fourier transform in 2D.
This is described in a paper by Colin J.R. Sheppard
"""
function hgoal_j0(rrel)
    rrel = min(one(typeof(rrel)), rrel)  # ensure rrel is at least one
    return besselj0(typeof(rrel)(first_bessel_crossing) * rrel)
end

"""
    hgoal_leaver_1D(rrel)
Monotonously decreasing function in 1D
In the 1975 paper by Leaver, the argument is given for a line spread function
"""
function hgoal_leaver_1D(rrel)
    return (rrel < 1) * (one(typeof(rrel)) - (rrel*3 - rrel^3)/2)
end

"""
    hgoal_leaver_2D(rrel)
This is a 2D version from the Leaver & Smith 1975 paper.
Yealds a monotonically decaying function in its Fourier transformation.
"""
function hgoal_leaver_2D(rrel)
    rrel = min(one(typeof(rrel)), rrel)  # ensure rrel is at least one
    rrel2 = abs2(rrel)
    rrel4 = abs2(rrel2)
    rrtmp = sqrt(max(0, 1 .- abs2(rrel)))
    return (rrel < 1) * (rrtmp * (8 + 10 * rrel2 - 3*rrel4) - 3*(8*rrel2 - 4*rrel4 + rrel4*rrel2) * log((1 + rrtmp) / rrel)) / 8
end

"""
    hgoal_lucosz_sufficient(rrel)
The lucosz sufficient function is a 2D function as described in the paper by Stallinga, Young, .
The function does approximate the Lucosz lower bound by a continuous function but is not really a "sufficient" condition for positivity,
and this yields small negative values.
"""
function hgoal_lucosz_sufficient(rrel)
    rrel = min(one(typeof(rrel)), rrel)  # ensure rrel is at least one
    return min(cos(pi.*rrel / (rrel + 1)), abs2(cos(pi*rrel/(rrel+sqrt(typeof(rrel)(2)))))) # 2D
end

"""
    hgoal_standard_otf(rrel)
This standard scalar OTF of a widefield system maximizes the Strehl ratio.
It is described in the paper by Stallinga, Enderlein et al.
"""
function hgoal_standard_otf(rrel)
    rrel = min(one(typeof(rrel)), rrel)  # ensure rrel is at least one
    return acos(1 - rrel)  # 2D
end

"""
    hgoal_hanning(rrel; rel_start=0.0)

The Hanning function yields a flat middle OTF that is apodized with a cosine function near its edge.
"""
function hgoal_hanning(rrel; rel_start=0.0)
    rrel = max(zero(typeof(rrel)), (rrel - rel_start)/(1 - rel_start))
    rrel = min(one(typeof(rrel)), rrel)  # ensure rrel is at least one
    return abs2(cos(pi*rrel))
end

"""
    hgoal_sonine(rrel, a=0.5)
The sonine function is a polynomial that is used in the 2D case.
This is described in a paper by C.J.R. Sheppard.
The following are sensible choices of a:
a: 0, 0.5, 1, 3-sqrt(3)
"""
function hgoal_sonine(rrel, a=0.5)
    rrel = min(one(typeof(rrel)), rrel)  # ensure rrel is at least one
    return (1- a) .+ a .* abs2.(rrel)
end

"""
    hgoal_one(rrel)

The hgoal_one function is a simple function that returns a constant value of one.
This is the default. The Wiener filter itself has the property to apodize.
Note that this is the only function that yields a non-zero value for rrel > 1.
Thus only this function can be used for reconstructions preserving high-frequency noise.
"""
function hgoal_one(rrel)
    return one(typeof(rrel))
end

"""
    hgoal_exp(rrel; exponent=0.5)
The hgoal_exp function is a generalization of the hgoal_one function.
It is used to create a hgoal function that is a power of (1 - rrel).
This is useful for creating a hgoal function that is more flexible than the hgoal_one function. 
"""
function hgoal_exp(rrel; exponent=typeof(rrel)(0.5))
    rrel = min(one(typeof(rrel)), rrel)  # ensure rrel is at least one
    return (1 .- rrel) .^ typeof(rrel)(exponent)
end