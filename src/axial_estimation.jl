# Estimation of the axial frequency k_z and the axial phase φ_z of axially
# modulated orders (three-beam 3-D SIM, lattices with a zero-order beam) from
# a 3-D stack.
#
# After unmixing with the (estimated) lateral parameters, order m is the object
# modulated laterally by K and imaged with h_m = h ⋅ cos(π k_z z + φ_z), and the
# zero order is the object imaged with h. Demodulating the order by K, in the
# region where both transfer functions are non-zero:
#     G_m(q) ⋅ OTF(q) = F_0(q) ⋅ c ⋅ FT[h ⋅ cos(π k_z z + φ_z) ⋅ e^{-iπK⋅x}](q)
# with one complex amplitude c. For given k_z this is linear in c, and with
# C = FT[h cos(π k_z z) e^{-iπK⋅x}], S = FT[h sin(...)] the model is
# c (cos φ_z C − sin φ_z S). k_z is searched around a seed (the geometric value),
# φ_z on a grid. In simulations φ_z is found to about ±0.05 rad and k_z to a
# few percent (a non-integer lateral K leaves a small model mismatch at the
# field edges); the reconstruction is insensitive to such k_z errors, but not
# to a wrong φ_z.

_ctr_coords(n) = collect(0:n-1) .- n ÷ 2

"""
    estimate_axial(sim_data, sp::SIMParams; orders=nothing, rel_range=0.5, kz_steps=41, phi_steps=180,
                   common=true, thresh=0.01)

Fit the axial frequency `k_z` (relative to the axial Nyquist frequency) and the
axial phase `φ_z` of the axially modulated orders of `sp` from the 3-D stack
`sim_data` (x, y, z, images; the grid of `sp`). `sp.mypsf` must be the 3-D PSF.

+ `orders`: peak indices to fit; default: all peaks with `k_peak_pos[i][3] != 0`
  (their value is the seed, e.g. from `get_kz` or `lattice_peaks(...; k0)`).
+ `rel_range`: k_z is searched within `seed ⋅ (1 ± rel_range)`, `kz_steps` values, then refined.
+ `common=true`: orders with the same seed share one fitted `k_z, φ_z` (one OTF);
  `false`: every order is fitted on its own (individual OTFs, e.g. for misaligned beams).

Returns `(sp_new, results)`: a copy of `sp` with the fitted axial frequencies in
`k_peak_pos` and phases in `otf_phases` (and `otf_indices` from `make_3d_pattern`),
and per fitted group a NamedTuple `(orders, kz, phi, residual)`.
"""
function estimate_axial(sim_data, sp::SIMParams; orders=nothing, rel_range=0.5, kz_steps=41, phi_steps=180,
                        common=true, thresh=0.01)
    sz = size(sim_data)[1:3]
    sz[3] > 1 || error("estimate_axial needs a 3-D stack (x, y, z, images).")
    N = size(sim_data, ndims(sim_data))
    size(sp.mypsf)[1:3] == sz || error("sp.mypsf ($(size(sp.mypsf))) must have the size of the stack ($sz).")
    P = length(sp.k_peak_pos)
    sel = isnothing(orders) ? [i for i in 2:P if sp.k_peak_pos[i][3] != 0] : collect(orders)
    isempty(sel) && return SIMParams(sp), NamedTuple[]
    I = reshape(ComplexF64.(Array(sim_data)), :, N)
    pm = pinv_weight_matrix(sp)
    h = Float64.(Array(sp.mypsf))
    x = reshape(_ctr_coords(sz[1]), :, 1, 1); y = reshape(_ctr_coords(sz[2]), 1, :, 1); z = reshape(_ctr_coords(sz[3]), 1, 1, :)
    F0 = fft(reshape(I * pm[1, :], sz))
    OTF0 = fft(ifftshift(h))
    o0 = abs.(OTF0) .> thresh * maximum(abs, OTF0)
    # per order: data in the overlap region and the model factors for a given k_z
    data = Dict{Int,Any}()
    for m in sel
        K = sp.k_peak_pos[m]
        dem = cis.(-pi .* (K[1] .* x .+ K[2] .* y))
        Gm = fft(reshape(I * pm[m, :], sz) .* dem)
        Ofs = fft(ifftshift(h .* dem))
        mask = o0 .& (abs.(Ofs) .> thresh * maximum(abs, Ofs))
        data[m] = (y=(Gm .* OTF0)[mask], F0=F0[mask], dem=dem, mask=mask)
    end
    seed(m) = abs(sp.k_peak_pos[m][3])
    groups = common ? [[m for m in sel if isapprox(seed(m), s; rtol=1e-9)] for s in unique(seed.(sel))] : [[m] for m in sel]
    # φ_z and φ_z + π give the same model with the amplitude negated: search
    # half a period, then pick the sign that makes the amplitudes positive (as
    # after unmixing with the right lateral phases).
    φs = range(-pi / 2, pi / 2; length=phi_steps + 1)[1:end-1]
    # residual (relative) of a group for k_z and every φ_z: min over each order's complex amplitude
    function residuals(grp, kz)
        r = zeros(length(φs)); tot = 0.0
        for m in grp
            d = data[m]
            C = fft(ifftshift(h .* cos.(pi * kz .* z) .* d.dem))[d.mask] .* d.F0
            S = fft(ifftshift(h .* sin.(pi * kz .* z) .* d.dem))[d.mask] .* d.F0
            CC = sum(abs2, C); SS = sum(abs2, S); CS = real(sum(conj.(C) .* S))
            Cy = sum(conj.(C) .* d.y); Sy = sum(conj.(S) .* d.y); yy = sum(abs2, d.y)
            for (j, φ) in enumerate(φs)
                c, s = cos(φ), sin(φ)
                MM = c^2 * CC + s^2 * SS - 2c * s * CS
                r[j] += yy - abs2(c * Cy - s * Sy) / max(MM, eps())
            end
            tot += yy
        end
        return r ./ tot
    end
    kp = copy(sp.k_peak_pos)
    phases = copy(sp.otf_phases)
    results = NamedTuple[]
    for grp in groups
        s0 = seed(first(grp))
        s0 == 0 && (s0 = 0.1)
        best = (Inf, s0, 0.0)
        consider(kz) = begin
            r = residuals(grp, kz)
            j = argmin(r)
            r[j] < best[1] && (best = (r[j], kz, φs[j]))
        end
        hgrid = 2 * rel_range * s0 / (kz_steps - 1)
        for kz in range(max(0.0, s0 * (1 - rel_range)), s0 * (1 + rel_range); length=kz_steps)
            consider(kz)
        end
        for _ in 1:3   # refine
            for kz in range(max(0.0, best[2] - hgrid), best[2] + hgrid; length=7)
                consider(kz)
            end
            hgrid /= 3
        end
        r, kz, φ = best
        amp = 0.0                     # summed real part of the orders' amplitudes c
        for m in grp
            d = data[m]
            M = (cos(φ) .* fft(ifftshift(h .* cos.(pi * kz .* z) .* d.dem))[d.mask] .-
                 sin(φ) .* fft(ifftshift(h .* sin.(pi * kz .* z) .* d.dem))[d.mask]) .* d.F0
            amp += real(sum(conj.(M) .* d.y) / max(sum(abs2, M), eps()))
        end
        amp < 0 && (φ = φ > 0 ? φ - pi : φ + pi)
        for m in grp
            kp[m] = (kp[m][1], kp[m][2], kz)
            phases[m] = φ
        end
        push!(results, (orders=grp, kz=kz, phi=φ, residual=r))
    end
    otf_indices, _ = make_3d_pattern(kp, 0.0)
    return SIMParams(sp; k_peak_pos=kp, otf_indices=otf_indices, otf_phases=phases), results
end
