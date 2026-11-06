# k-vector and phase estimation of 3-D SIM stacks by an OTF-matched 3-D correlation.
#
# In a z-stack every (axially modulated) order m is the object, modulated laterally by K, imaged with its own OTF
# OTF_m = FT[h ⋅ cos(π k_z z + φ_z)] (OTF_0 = FT[h] for the zero order and lateral orders). A single plane of a thick
# sample is dominated by out-of-focus light, and the z-sum (k_z = 0) of an axially modulated order is nearly empty
# (its two copies at ±k_z barely reach k_z = 0), so a 2-D estimate on a plane or on the z-average locates such orders
# poorly. Here every image is filtered with conj(OTF_m) and the direction's mean (the zero order) with conj(OTF_0);
# their product summed over z is (Parseval along z) the 3-D cross-correlation at Δk_z = 0:
#     P_n(x, y) = Σ_z f_n(x, y, z) ⋅ g(x, y, z),   c_n(K) = Σ_xy P_n(x, y) e^{-iπ K⋅x}
# (K relative to the Nyquist frequency, x in pixels from the centre as in `SIMPattern`). K maximizes Σ_n |c_n(K)|² over
# the images of a direction; the phase of image n is arg c_n(K). Every plane contributes and the haze outside the
# order's OTF support is filtered out.

# |c_n|² summed over the images on a grid of peak positions around K (pixels from DC), by two matrix products per
# image (a zoomed DFT): returns the best position on the grid.
function _best_on_grid(P, K, offs)
    nx, ny = size(P, 1), size(P, 2)
    xs = collect(0:nx-1) .- nx ÷ 2
    ys = collect(0:ny-1) .- ny ÷ 2
    Ex = cis.(-2pi .* ((K[1] .+ offs) .* transpose(xs)) ./ nx)     # (grid, x)
    Ey = cis.(-2pi .* (ys .* transpose(K[2] .+ offs)) ./ ny)       # (y, grid)
    S = zeros(real(eltype(P)), length(offs), length(offs))
    for p in axes(P, 3)
        S .+= abs2.(Ex * view(P, :, :, p) * Ey)
    end
    i = argmax(S)
    return (K[1] + offs[i[1]], K[2] + offs[i[2]])
end

# c_n(K) of every image
function _corr_at(P, K)
    nx, ny = size(P, 1), size(P, 2)
    ex = cis.(-2pi .* K[1] .* (collect(0:nx-1) .- nx ÷ 2) ./ nx)
    ey = cis.(-2pi .* K[2] .* (collect(0:ny-1) .- ny ÷ 2) ./ ny)
    return [transpose(ex) * view(P, :, :, p) * ey for p in axes(P, 3)]
end

"""
    estimate_parameters_3d(dat, psf; k_vecs, num_directions, kz=0.0, otf_phase=0.0, implied_higher_orders=0,
                           search=2.0, ideal_strength=true, verbose=false)

Estimate the lateral k-vectors and the phases of a 3-D SIM stack `dat` (x, y, z, images) by an OTF-matched 3-D
correlation (see the comment at the top of this file). Compared with `estimate_parameters` on one plane or on the
z-average, every plane contributes and out-of-focus light is suppressed, which matters most for axially modulated orders
(three-beam 3-D SIM), whose OTF copies at ±k_z hardly overlap k_z = 0.

+ `psf`: the 3-D PSF on the grid of the stack (centred).
+ `k_vecs`: per direction the start position of its first order in pixels from DC (as for `estimate_parameters`).
+ `num_directions`: the number of directions (images split equally), or a vector of the images per direction.
+ `kz`: the axial frequency of the first order (relative to the axial Nyquist frequency; 0 = laterally only modulated),
  one number or one per direction. It only shapes the matched filter: an error of 20 % changes little.
  `otf_phase`: its axial phase.
+ `implied_higher_orders`: higher orders implied from the first (k ⋅ (h+1), phase ⋅ (h+1)); a number or one per direction.
+ `search`: the k-vectors are searched within ± `search` pixels of the start, then refined to 0.001 pixels.
+ `ideal_strength`: strengths 0.5 (as `estimate_parameters`); `false`: 0.5 times the measured correlation magnitude
  relative to its mean over the direction's images (the per-image variations).

Returns a `SIMParams` with the layout of `estimate_parameters(...; num_directions, implied_higher_orders)`: the zero
order, then per direction its first order and the implied ones (axial components 0; fit them with `estimate_axial`).
"""
function estimate_parameters_3d(dat, psf; k_vecs, num_directions, kz=0.0, otf_phase=0.0, implied_higher_orders=0,
                                search=2.0, ideal_strength=true, verbose=false)
    ndims(dat) == 4 || error("estimate_parameters_3d needs a stack (x, y, z, images).")
    vol = size(dat)[1:3]
    size(psf)[1:3] == vol || error("The PSF ($(size(psf))) must have the size of the stack ($vol).")
    N = size(dat, 4)
    rngs = direction_ranges(N, num_directions)
    D = length(rngs)
    length(k_vecs) >= D || error("Provide one start k-vector per direction.")
    per(v, d) = v isa AbstractVector ? v[d] : v
    T = Float32
    h = T.(Array(psf))
    h ./= sum(h)
    z = reshape(T.(collect(0:vol[3]-1) .- vol[3] ÷ 2), 1, 1, :)
    OTF0 = fft(ifftshift(h))
    otf_cache = Dict{Tuple{Float64,Float64},Array{Complex{T},3}}()
    otf_of(k, φ) = k == 0 ? OTF0 : get!(otf_cache, (Float64(k), Float64(φ))) do
        fft(ifftshift(h .* cos.(T(pi * k) .* z .+ T(φ))))
    end
    k_peak_pos = [(0.0, 0.0, 0.0)]
    phase_cols = Vector{Vector{Float64}}()
    strength_cols = Vector{Vector{Float64}}()
    for d in 1:D
        idx = rngs[d]
        sub = T.(Array(view(dat, :, :, :, idx)))
        m = dropdims(mean(sub; dims=4); dims=4)
        g = real.(ifft(fft(m) .* conj.(OTF0)))
        Om = otf_of(per(kz, d), per(otf_phase, d))
        sm = sum(m)
        P = Array{Complex{T}}(undef, vol[1], vol[2], length(idx))
        for (j, _) in enumerate(idx)
            dn = view(sub, :, :, :, j) .- m .* (sum(view(sub, :, :, :, j)) / sm)
            f = ifft(fft(dn) .* conj.(Om))
            P[:, :, j] = dropdims(sum(f .* g; dims=3); dims=3)
        end
        K = (Float64(k_vecs[d][1]), Float64(k_vecs[d][2]))
        for (r, s) in ((Float64(search), 0.1), (0.15, 0.01), (0.015, 0.001))
            K = _best_on_grid(P, K, collect(-r:s:r))
        end
        c = _corr_at(P, K)
        verbose && println("direction $d: k = $(round.(K; digits=3)) px")
        kr = (2 * K[1] / vol[1], 2 * K[2] / vol[2])        # relative to Nyquist
        nimp = per(implied_higher_orders, d)
        mags = abs.(c)
        mm = sum(mags) / length(mags)
        for o in 1:1+nimp
            push!(k_peak_pos, (o * kr[1], o * kr[2], 0.0))
            ph = zeros(N); st = zeros(N)
            ph[idx] .= o .* angle.(c)
            st[idx] .= (o == 1 && !ideal_strength && mm > 0) ? 0.5 .* mags ./ mm : 0.5
            push!(phase_cols, ph); push!(strength_cols, st)
        end
    end
    P = length(k_peak_pos)
    peak_phases = zeros(N, P)
    peak_strengths = zeros(N, P)
    peak_strengths[:, 1] .= 0.5
    for i in 2:P
        peak_phases[:, i] .= phase_cols[i-1]
        peak_strengths[:, i] .= strength_cols[i-1]
    end
    peak_strengths ./= maximum(peak_strengths)
    return SIMParams(Array(psf), k_peak_pos, peak_phases, peak_strengths, ones(Int, P), zeros(P))
end
