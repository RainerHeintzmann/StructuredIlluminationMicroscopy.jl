"""
Non-linear SIM illumination schemes.

In several non-linear schemes the non-linearity acts on the ILLUMINATION, not on the emission: the emitted signal is the object
times a periodic effective pattern `E(v)` with `v = k·r + φ` (the phase of the pattern, `φ` the phase step of the image).
`E` is then a sum of harmonics, which is exactly what a `SIMParams` pattern represents (`SIMPattern`: `Σ strength·cos(k_n·r + phase)`),
so such a scheme only needs the harmonic strengths and phase offsets `E(v) = c_0 + Σ_m 2|c_m| cos(m v + arg c_m)`.
`generate_peaks` / `generate_peaks_directions` take a `scheme` and put them into the peak strengths and phases
(the simulation then runs with `emission_modification = identity`).
"""
abstract type NonlinearScheme end

"""
    ActivationDepletionReadout(; act_sat=1.0, act_mod=1.0, act_shift=0.0,
                                 depl_sat=8.0, depl_mod=1.0, read_mod=1.0, read_shift=0.0)

Positive-contrast scheme with three stages, all with the same period (phase `v`; the dark point of the depletion pattern is at `v = 0`):

1. **Activation** with the pattern `I_a(v) = act_sat · (1 + act_mod · cos(v − act_shift))` (intensity in units of the saturation
   intensity of the activation): activated fraction `A = 1 − exp(−I_a)`. `act_mod = 0` is a flat (widefield) activation.
2. **Saturated depletion** with `I_d(v) = depl_sat · (1 − depl_mod · cos v)` (mean intensity in units of the saturation intensity of
   the depletion): surviving fraction `S = exp(−I_d)`. For `depl_sat ≫ 1` only narrow regions around the depletion zeros
   survive; their width, hence the strength of the higher harmonics, shrinks with `depl_sat`.
3. **Readout** with `R(v) = 1 + read_mod · cos(v − read_shift)`, shifted by `read_shift` (radian) from the depletion zero.

The emitted signal is `E(v) = A · S · R`. Angles are in radian; all intensities are relative, so `E` is only defined up to a factor.
"""
Base.@kwdef struct ActivationDepletionReadout <: NonlinearScheme
    act_sat::Float64 = 1.0
    act_mod::Float64 = 1.0
    act_shift::Float64 = 0.0
    depl_sat::Float64 = 8.0
    depl_mod::Float64 = 1.0
    read_mod::Float64 = 1.0
    read_shift::Float64 = 0.0
end

"""
    PeriodicProfile(f)

A scheme given by its effective pattern `f(v)` (`v` in radian, period `2π`, real, not all zero), e.g. measured or from another model.
"""
struct PeriodicProfile{F} <: NonlinearScheme
    f::F
end

"""
    scheme_profile(scheme, v)

The effective periodic pattern `E(v)` of the scheme (unnormalized).
"""
function scheme_profile(s::ActivationDepletionReadout, v)
    A = 1 - exp(-s.act_sat * (1 + s.act_mod * cos(v - s.act_shift)))
    S = exp(-s.depl_sat * (1 - s.depl_mod * cos(v)))
    R = 1 + s.read_mod * cos(v - s.read_shift)
    return A * S * R
end
scheme_profile(s::PeriodicProfile, v) = s.f(v)

"""
    scheme_coefficients(scheme, M; N=1024)

The complex Fourier coefficients `r_m = c_m / c_0`, `m = 0 … M`, of the effective pattern `E(v) = Σ_m c_m e^{i m v}` (so `r_0 = 1`).
The order `m` of the illumination then has the strength `2|r_m|` and the phase offset `arg r_m` relative to the zero order
(`E = c_0 [1 + Σ_{m≥1} 2|r_m| cos(m v + arg r_m)]`). `N` samples per period (must exceed `2M`).
"""
function scheme_coefficients(scheme::NonlinearScheme, M::Integer; N::Integer=1024)
    N > 2M || throw(ArgumentError("N = $N samples per period are not enough for $M harmonics"))
    E = [Float64(scheme_profile(scheme, 2π * (n - 1) / N)) for n in 1:N]
    all(isfinite, E) || throw(ArgumentError("the pattern of the scheme is not finite"))
    c = fft(E) ./ N
    c0 = real(c[1])
    abs(c0) > 0 || throw(ArgumentError("the mean of the effective pattern is zero"))
    return ComplexF64[c[m + 1] / c0 for m in 0:M]
end

"""
    scheme_strengths(scheme, num_orders)

The strengths `2|r_m|` and phase offsets `arg r_m` of the orders `m = 0 … num_orders-1` (the zero order has strength 1, phase 0),
as used for `peak_strengths` / `peak_phases`. Returns `(strengths, phases)`.
"""
function scheme_strengths(scheme::NonlinearScheme, num_orders::Integer)
    r = scheme_coefficients(scheme, num_orders - 1)
    strengths = [m == 0 ? 1.0 : (2 * abs(r[m + 1]) < 1e-9 ? 0.0 : 2 * abs(r[m + 1])) for m in 0:num_orders-1]   # (numerically zero orders are absent)
    phases = [m == 0 ? 0.0 : angle(r[m + 1]) for m in 0:num_orders-1]
    return strengths, phases
end

"""
    scheme_truncation(scheme, num_orders; N=1024)

The fraction of the (mean-free) power of the effective pattern beyond the harmonics `0 … num_orders-1`: how much a simulation
or reconstruction with `num_orders` orders neglects.
"""
function scheme_truncation(scheme::NonlinearScheme, num_orders::Integer; N::Integer=1024)
    M = N ÷ 2 - 1
    r = scheme_coefficients(scheme, M; N=N)
    total = sum(abs2, r[2:end])
    total > 0 || return 0.0
    return sum(abs2, r[num_orders + 1:end]) / total
end
