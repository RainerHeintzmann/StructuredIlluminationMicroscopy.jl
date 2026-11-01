"""
Model-free estimation of the strengths and phase offsets of the higher illumination orders (non-linear SIM).

The orders of one direction are separated from the images by the pseudo-inverse of the weight matrix built from the (assumed)
phases alone. A constant phase error of an order then just rotates its separated image (the strength never matters for the
separation), so it is recovered together with the strength. Neighbouring orders overlap in Fourier space by the shifted OTFs:
with `X_m(q) = c_m O(q - k_m) H(q)` (the separated order `m`, `c_m` the complex coefficient of the illumination)

    X_m(q) H(q - δ_m)  =  (c_m / c_{m-1}) · X_{m-1}(q - δ_m) H(q),    δ_m = k_m - k_{m-1},

whatever the object `O` is. A least-squares fit over the overlap gives the complex ratio `r_m = c_m / c_{m-1}` of every pair of
neighbours, and the cumulative products `R_m = c_m / c_0`. This needs only `|δ_m| < 2·cutoff`, not `|k_m| < 2·cutoff` as a fit
against the zero order does, so orders far beyond the support of the zero order are measurable too.

`H` is the OTF of the in-focus plane (a 3-D PSF is reduced to its centre plane). For a single 2-D slice of a 3-D sample the
out-of-focus haze sits at low frequencies around every order's centre, (detection frequency) `q = 0` of every separated order, i.e. at `q ≈ 0` and `q ≈ δ_m` in the fit below. The identity above needs the true
OTF, so the notch of the reconstruction (`gaussian_notch`: `1 - c·exp(-|q|²/2σ²)`, `q` in cycles per pixel) is applied as a smooth
least-squares WEIGHT `n(q) n(q - δ_m)` instead: the fit is confocal-like, dominated by the in-focus information away from
the order centres, without biasing the ratio.

Noise in `B` would bias the least-squares ratio towards zero (the denominator `Σ|B|²` contains the noise power). Above the cutoff of
the OTF the separated orders hold only noise, which gives the noise power of every order, and its expected contribution is subtracted
from `Σ|A|²` and `Σ|B|²` (`noise_correction`); the cross term `Σ A B*` is unbiased since the noise of different orders is independent.

The haze is not removed by the notch alone: an out-of-focus plane obeys the same relation, but with its defocused OTF `H_z` instead of
`H`, and the pairing `H_z(q) H(q - δ)` against `H_z(q - δ) H(q)` is consistent only where the attenuation of `q` and `q - δ` by the
defocus agrees. For a rotationally symmetric PSF the defocus attenuation depends on `|q|` only, so the haze cancels exactly on the
bisector `|q| = |q - δ|` of the overlap lens. The fit is therefore also weighted by `exp(-(|q| - |q - δ|)² / 2τ²)`, `τ =
haze_symmetry · cutoff` (`haze_symmetry = 0` switches it off; it costs overlap area, so it is only worth it for thick samples).
"""

# raised-cosine taper at the image borders (the subpixel shifts by phase ramps do not leak across the periodic boundary)
function _edge_taper(n, frac=0.1)
    m = max(1, round(Int, frac * n))
    return [min(1.0, sin(pi / 2 * min(j, n - 1 - j) / m)^2 + (min(j, n - 1 - j) >= m)) for j in 0:n-1]
end

# the centred 2-D PSF of the in-focus plane with unit sum
function _inplane_psf(mypsf)
    p = ndims(mypsf) == 3 ? mypsf[:, :, size(mypsf, 3) ÷ 2 + 1] : mypsf
    p = Float64.(collect(p))
    s = sum(p)
    s > 0 || error("The PSF sums to zero.")
    return p ./ s
end

# noise power of a separated order image: the mean power beyond the OTF cutoff `kc` (bins), where the signal vanishes; the corners
# of the square spectrum count (a wide OTF leaves no ring inside the Nyquist circle), the outermost bins are left out
function _band_noise(Xj, qx, qy, qr, kc, nx, ny)
    band = (qr .> 1.15 * kc) .& (abs.(qx) .< 0.95 * nx / 2) .& (abs.(qy) .< 0.95 * ny / 2)
    count(band) >= 50 ? sum(abs2, Xj[band]) / count(band) : 0.0
end

# groups of peaks that share their images (one direction each): (images, peaks ordered by |k|)
function _order_groups(sp::SIMParams, present)
    P = length(sp.k_peak_pos)
    groups = Dict{Vector{Int},Vector{Int}}()
    for i in 2:P
        any(present[:, i]) || continue
        push!(get!(groups, findall(present[:, i]), Int[]), i)
    end
    out = NamedTuple{(:images, :peaks),Tuple{Vector{Int},Vector{Int}}}[]
    for (imgs, peaks) in sort(collect(groups); by=g -> first(g)[1])
        k1 = sp.k_peak_pos[first(peaks)]
        for i in peaks
            k = sp.k_peak_pos[i]
            abs(k1[1] * k[2] - k1[2] * k[1]) <= 1e-6 * (hypot(k1[1], k1[2]) * hypot(k[1], k[2]) + 1e-12) ||
                error("The peaks of the images $(imgs) are not collinear (lattice illumination?): the order chain needs gratings.")
        end
        push!(out, (images=imgs, peaks=sort(peaks; by=i -> hypot(sp.k_peak_pos[i][1], sp.k_peak_pos[i][2]))))
    end
    return out
end

"""
    estimate_order_strengths(dat, sp, mypsf=sp.mypsf; notch_contrast=1.0, notch_sigma=0.05, haze_symmetry=0.0, noise_correction=true, max_noise_fraction=0.7, taper=0.1, iterations=2, min_overlap=0.002, min_coherence=0.1)

Estimate the strengths and the phase offsets of all orders `m ≥ 1` of every direction of `sp` from the images `dat` (`[x, y, images]`,
the rows of `sp.peak_phases`) without a model of the non-linearity, see the header of this file. `sp` provides the k-vectors, the
(assumed) phases, e.g. `m` times the phase of the first order, and which images contain which peaks.

Returns `(sp_new, info)`: a copy of `sp` with `peak_strengths` and `peak_phases` of the orders ≥ 1 replaced by the measured values
(the strength relative to the zero order of the same image, i.e. `2|c_m/c_0|` times `sp.peak_strengths[:, 1]`; orders that could
not be measured get the strength 0; the phase of order 1 is kept, the higher orders get the deviation of their phase from the linear
progression `m·arg R_1`), and a vector with one entry per direction: `(images, peaks, R, r, phase_dev, coherence, overlap, valid)`,
the cumulative `R_m = c_m/c_0`, the neighbour ratios `r_m`, the phase deviations from the linear progression, the fit coherence `|Σ A B*|² / (Σ|A|² Σ|B|²)`, the overlap area as a
fraction of the image and whether the order was measured.

Parameters:
+ `mypsf` : PSF (2-D, or 3-D: its centre plane is used) at the sampling of the images
+ `notch_contrast`, `notch_sigma` : the Gaussian notch (as `ReconParams.suppression_strength` and `suppression_sigma`, `σ` in cycles per
  pixel) that weights the fit around the order centres (see above); `notch_contrast = 0` weights uniformly and excludes a small disc
  around the order centres instead
+ `haze_symmetry` : relative width `τ / cutoff` of the symmetry weight on the bisector of the overlap (see above), 0 = off; try 0.1–0.3
  for a slice of a thick sample
+ `noise_correction` : subtract the expected noise power from the sums of the fit (see above); needs some of the spectrum (the corners
  of the square) beyond the OTF cutoff to estimate the noise power from; without it the correction is skipped
+ `max_noise_fraction` : an order whose fit region holds more than this fraction of noise power (after the expected noise was determined from the
  band beyond the cutoff) is not measurable; the correction cannot be trusted there. The coherence must lie in `[min_coherence, 1.3]` (above 1 only the noise correction can account for)
+ `taper` : the width of the raised-cosine edge taper as a fraction of the image size. The taper smooths the spectra by about 1/`taper`
  bins, which must stay small against the width of the overlap region: small images need a narrow taper (but the wrap-around
  discontinuity of the unwindowed borders leaks the strong low frequencies), large images are not sensitive
+ `iterations` : the separation of the orders uses the assumed phases; every pass re-separates with the phase offsets measured by the
  previous one (the first pass already gives them to first order, a further pass removes the remaining crosstalk between the orders)
+ `min_overlap` : the minimum overlap area (fraction of the image) of the two shifted OTFs
+ `min_coherence` : a lower bound of the fit coherence; below it the order (and the higher ones) count as not measurable
"""
function estimate_order_strengths(dat::AbstractArray{<:Real,3}, sp::SIMParams, mypsf=sp.mypsf;
                                  notch_contrast=1.0, notch_sigma=0.05, haze_symmetry=0.0, noise_correction=true, max_noise_fraction=0.7, taper=0.1, iterations=2, min_overlap=0.002, min_coherence=0.1)
    nx, ny, N = size(dat)
    size(sp.peak_phases, 1) == N || throw(DimensionMismatch("dat has $N images, sp.peak_phases has $(size(sp.peak_phases, 1)) rows."))
    mypsf === nothing && error("estimate_order_strengths needs the PSF.")
    present = sp.peak_strengths .!= 0
    groups = _order_groups(sp, present)

    hc = _inplane_psf(mypsf)
    size(hc) == (nx, ny) || throw(DimensionMismatch("The PSF has the size $(size(hc)), the images $((nx, ny))."))
    H = fft(ifftshift(hc))
    x = 0:nx-1
    y = (0:ny-1)'
    xc = x .- nx ÷ 2
    yc = y .- ny ÷ 2
    qx = fftfreq(nx, nx)                                  # in bins
    qy = fftfreq(ny, ny)'
    notch(a, b) = 1 .- notch_contrast .* exp.(-((a ./ nx) .^ 2 .+ (b ./ ny) .^ 2) ./ (2 * notch_sigma^2))   # (bins → cycles/pixel)
    r0 = notch_contrast > 0 ? 0.0 : max(2.0, 0.02 * min(nx, ny))
    qr = hypot.(qx, qy)
    kc = maximum(qr[abs.(H) .> 1e-3 * maximum(abs.(H))])      # the cutoff of the OTF in bins
    win = _edge_taper(nx, taper) .* _edge_taper(ny, taper)'
    F = fft(Float64.(dat) .* win, (1, 2))

    strengths = copy(sp.peak_strengths)
    phases = copy(sp.peak_phases)
    info = NamedTuple[]
    for g in groups
        I, J = g.images, g.peaks
        M, L = length(J), length(g.images)
        Meff = min(M, (L - 1) ÷ 2)                       # 2M + 1 columns need at least that many images
        R = fill(complex(NaN), M)
        r = fill(complex(NaN), M)
        coh = fill(NaN, M)
        ovl = fill(NaN, M)
        valid = falses(M)
        dev = zeros(M)                                    # the phase offsets found so far (deviation from the linear progression)
        for pass in 1:(Meff >= 1 ? iterations : 0)
            fill!(R, complex(NaN)); fill!(r, complex(NaN)); fill!(coh, NaN); fill!(ovl, NaN); fill!(valid, false)
            wm = ComplexF64[cis(sp.peak_phases[i, j] + dev[m]) for i in I, (m, j) in enumerate(J[1:Meff])]
            W = hcat(ones(ComplexF64, L), wm, conj.(wm))
            X = pinv(W) * transpose(reshape(F[:, :, I], nx * ny, L))
            noisepow = [noise_correction ? _band_noise(reshape(X[j, :], nx, ny), qx, qy, qr, kc, nx, ny) : 0.0 for j in 1:size(X, 1)]
            xprev = ifft(reshape(X[1, :], nx, ny))      # X_0 in real space
            kprev = (0.0, 0.0)
            Rprev = 1.0 + 0.0im
            for m in 1:Meff
                kk = sp.k_peak_pos[J[m]]
                k = (kk[1] * nx / 2, kk[2] * ny / 2)
                δ = (k[1] - kprev[1], k[2] - kprev[2])
                ramp = cis.(2pi .* (δ[1] .* x ./ nx .+ δ[2] .* y ./ ny))
                Hδ = fft(ifftshift(hc .* cis.(2pi .* (δ[1] .* xc ./ nx .+ δ[2] .* yc ./ ny))))   # H(q - δ)
                B = fft(xprev .* ramp) .* H                                                     # X_{m-1}(q - δ) H(q)
                Xm = reshape(X[1 + m, :], nx, ny)
                A = Xm .* Hδ                                                                    # X_m(q) H(q - δ)
                w = abs.(H) .* abs.(Hδ)
                mask = (w .> 1e-3 * maximum(w))
                # the out-of-focus haze of X_m sits at the detection frequency q ≈ 0, that of the shifted X_{m-1}(q - δ) at q ≈ δ: the notch weights both down
                wq = notch_contrast > 0 ? notch(qx, qy) .* notch(qx .- δ[1], qy .- δ[2]) : 1.0
                haze_symmetry > 0 && (wq = wq .* exp.(-((qr .- hypot.(qx .- δ[1], qy .- δ[2])) ./ (haze_symmetry * kc)) .^ 2 ./ 2))
                if r0 > 0
                    mask .&= (hypot.(qx, qy) .> r0) .& (hypot.(qx .- δ[1], qy .- δ[2]) .> r0)
                end
                ovl[m] = count(mask) / (nx * ny)
                ovl[m] >= min_overlap || break
                Am, Bm = (A .* wq)[mask], (B .* wq)[mask]
                # expected noise power in the sums (A holds the noise of X_m, B that of X_{m-1})
                sB, sA = sum(abs2, Bm), sum(abs2, Am)
                den = sB - noisepow[m] * sum(abs2, (H .* wq)[mask])
                denA = sA - noisepow[1 + m] * sum(abs2, (Hδ .* wq)[mask])
                # An order whose power is mostly noise cannot be corrected reliably (a small error of the noise estimate is amplified):
                # not measurable, rather than a wrong number.
                (den > (1 - max_noise_fraction) * sB && denA > (1 - max_noise_fraction) * sA) || break
                num = sum(Am .* conj.(Bm))
                r[m] = num / den
                coh[m] = abs2(num) / (denA * den)
                (min_coherence <= coh[m] <= 1.3) || break
                R[m] = Rprev * r[m]
                valid[m] = true
                xprev = ifft(Xm)
                kprev = k
                Rprev = R[m]
            end
            valid[1] || break
            # The phase of order 1 stays as given (it fixes the position of the pattern, in the convention of whoever estimated it; the
            # array origin of this fit differs by a constant ε that would enter order m as m·ε); the higher orders get the deviation of
            # their phase from the linear progression m·arg R_1 (the phase offsets of a non-linear pattern).
            dev .+= [valid[m] ? angle(R[m] * conj(R[1])^m) : 0.0 for m in 1:M]
        end
        for (m, j) in enumerate(J)
            for p in I
                if valid[m]
                    strengths[p, j] = 2 * abs(R[m]) * sp.peak_strengths[p, 1]
                    phases[p, j] = mod(sp.peak_phases[p, j] + dev[m], 2pi)
                else
                    strengths[p, j] = 0.0
                end
            end
        end
        push!(info, (images=I, peaks=J, R=R, r=r, phase_dev=dev, coherence=coh, overlap=ovl, valid=valid))
    end
    sp_new = SIMParams(sp.mypsf, copy(sp.k_peak_pos), phases, strengths, copy(sp.otf_indices), copy(sp.otf_phases), sp.otf_exponent)
    return sp_new, info
end
