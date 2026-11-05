# Smooth distortions of the illumination pattern (`PatternDistortion`, see sim_structures.jl): a polynomial local phase
# ψ_i(x, y) per peak. `estimate_distortion` finds it together with the k-vectors by maximizing the correlation-peak
# strengths (the per-image phases then follow in closed form, as in `estimate_parameters`); the reconstruction removes it
# from every separated order (`distortion_phasors`, `separate_and_place_orders`).

"""
    distortion_terms(degree)

The Legendre-product terms `(a, b)` (`P_a(u) P_b(v)`, `a + b ≤ degree`) of a `PatternDistortion`, ordered by total degree.
"""
distortion_terms(degree) = [(a, t - a) for t in 0:degree for a in t:-1:0]

# Legendre polynomials P_0 … P_n at the points u: (n+1, length(u))
function _legendre(n, u)
    L = ones(Float64, n + 1, length(u))
    n >= 1 && (L[2, :] .= u)
    for j in 1:n-1
        L[j+2, :] .= ((2j + 1) .* u .* L[j+1, :] .- j .* L[j, :]) ./ (j + 1)
    end
    return L
end

# the normalized centred coordinate of a grid axis of length n: (x - (n÷2+1)) / (n/2) ∈ [-1, 1)
_norm_axis(n) = ((1:n) .- (n ÷ 2 + 1)) ./ (n / 2)

# Σ_t c[t] P_a(u) P_b(v) on the grid sz (separable: (deg+1)² products of vectors)
function _eval_legendre(c::AbstractVector, degree, sz)
    Lu, Lv = _legendre(degree, _norm_axis(sz[1])), _legendre(degree, _norm_axis(sz[2]))
    C = zeros(degree + 1, degree + 1)
    for (t, (a, b)) in enumerate(distortion_terms(degree))
        C[a+1, b+1] = c[t]
    end
    return Lu' * C * Lv
end

"""
    distortion_phase(sp::SIMParams, i, sz)

The local phase `ψ_i` (radians, size `sz[1:2]`) of peak `i` of a distorted pattern (`sp.distortion`), or `nothing`.
"""
function distortion_phase(sp::SIMParams, i, sz)
    d = sp.distortion
    (d === nothing || i > size(d.coefs, 1) || all(iszero, @view d.coefs[i, :])) && return nothing
    return _eval_legendre(@view(d.coefs[i, :]), d.degree, sz)
end

"""
    distortion_phasors(sim_data, sp::SIMParams)

Per order (peak) of `sp`, the factor `e^{-iψ_n}` (x, y; array type of `sim_data`) that removes the local phase of a distorted
pattern from the separated order; empty without a distortion. Computed once per preparation (one `cis` per pixel and order).
"""
function distortion_phasors(sim_data, sp::SIMParams)
    sp.distortion === nothing && return Any[]
    sz = size(sim_data)[1:2]
    CT = complex(eltype(sim_data))
    # (the separated order n carries e^{iψ_n(x)} of the pattern, checked against simulated distortions in the tests)
    return Any[(ψ = distortion_phase(sp, n, sz); ψ === nothing ? nothing : copyto!(similar(sim_data, CT, sz), cis.(.-ψ)))
               for n in eachindex(sp.k_peak_pos)]
end

function _apply_distortion_phasor!(order, prep, n)
    (!prep.use_distortion || isempty(prep.distortion_phasors) || n > length(prep.distortion_phasors)) && return order
    ph = prep.distortion_phasors[n]
    ph === nothing && return order
    order .*= reshape(ph, size(ph)..., ntuple(_ -> 1, ndims(order) - 2)...)     # (the same for every plane)
    return order
end

"""
    distortion_masks(sim_data, sp::SIMParams, otfs; kappa=1.0)

Per order of a distorted pattern, the part of its OTF support that the reconstruction keeps (Fourier, order grid; `nothing`
for undistorted orders). Removing the local phase after the blur (`distortion_phasors`) leaves the OTF at the locally
shifted frequency: the relative error is about `δ |∇OTF| / |OTF|` (δ: the largest local frequency shift `|∇ψ| n / 2π` in
Fourier pixels), large at the edge of the OTF, where the final filter amplifies it. Kept: `|OTF| > kappa δ |∇OTF|`.
"""
function distortion_masks(sim_data, sp::SIMParams, otfs; kappa=1.0)
    sp.distortion === nothing && return Any[]
    sz = size(sim_data)[1:2]
    RT = real(eltype(sim_data))
    masks = Any[]
    for n in eachindex(otfs)
        ψ = n <= length(sp.k_peak_pos) ? distortion_phase(sp, n, sz) : nothing
        if ψ === nothing
            push!(masks, nothing); continue
        end
        gx = (circshift(ψ, (-1, 0)) .- circshift(ψ, (1, 0))) ./ 2      # rad per pixel (ψ is smooth: no wrapping)
        gy = (circshift(ψ, (0, -1)) .- circshift(ψ, (0, 1))) ./ 2
        inner = (3:sz[1]-2, 3:sz[2]-2)
        δ = maximum(hypot.(gx[inner...] .* sz[1], gy[inner...] .* sz[2])) / 2π     # Fourier pixels
        a = abs.(Array(otfs[n]))
        a2 = ndims(a) > 2 ? a[:, :, (size(a, 3) + 1) ÷ 2] : a                   # (the in-focus section)
        ax = (circshift(a2, (-1, 0)) .- circshift(a2, (1, 0))) ./ 2
        ay = (circshift(a2, (0, -1)) .- circshift(a2, (0, 1))) ./ 2
        m = RT.((a2 .> kappa * δ .* hypot.(ax, ay)) .& (a2 .> 1e-3 * maximum(a2)))
        push!(masks, copyto!(similar(sim_data, RT, size(m)), m))
    end
    return masks
end

# after the FFT of a compensated order: back to the reliable part of its OTF support (`distortion_masks`)
function _mask_distorted_order!(order, prep, n)
    (!prep.use_distortion || n > length(prep.distortion_masks) || prep.distortion_masks[n] === nothing) && return order
    m = prep.distortion_masks[n]
    order .*= reshape(m, size(m)..., ntuple(_ -> 1, ndims(order) - ndims(m))...)
    return order
end

"""
    peak_groups(sp::SIMParams; tol=1e-6)

The peaks (indices into `sp.k_peak_pos`, without the zero order) grouped by their lateral direction: parallel k-vectors
(one SIM direction and its harmonics) form a group, sorted by |k| (the base peak first).
"""
function peak_groups(sp::SIMParams; tol=1e-6)
    groups = Vector{Int}[]
    for i in eachindex(sp.k_peak_pos)
        k = sp.k_peak_pos[i][1:2]
        nk = hypot(k...)
        nk > tol || continue
        g = findfirst(gr -> (q = sp.k_peak_pos[gr[1]][1:2]; abs(k[1] * q[2] - k[2] * q[1]) <= tol * nk * hypot(q...)), groups)
        g === nothing ? push!(groups, [i]) : push!(groups[g], i)
    end
    foreach(gr -> sort!(gr; by=i -> hypot(sp.k_peak_pos[i][1:2]...)), groups)
    return groups
end
_peak_group_of(sp::SIMParams, i) = findfirst(g -> i in g, peak_groups(sp))

"""
    estimate_distortion(dat, sp::SIMParams, mypsf=sp.mypsf; degree=3, model=:displacement, reg=1e-4, crop=64,
                        suppress_sigma=0.15, subtract_mean=true, otf_exponent=1.0, otf_moebius=1.0, iterations=300,
                        verbose=false) -> (sp_new, info)

Estimates a smooth distortion of the illumination pattern together with the k-vectors, starting from the parameters `sp`
(e.g. of `estimate_parameters`) for the images `dat` ((x, y, images) or (x, y, 1, images)).

As the k-vector refinement, it maximizes the correlation-peak strengths `Σ_m |c_m|²` of every image with the mean image
(the same PSF-filtered correlation as `estimate_parameters`), but with `e^{-ik·x}` replaced by `e^{-i(k·x + ψ(x))}`:
`c_m = Σ_x P_m(x) e^{-i(k·x + ψ(x))}`. k and the polynomial ψ are fitted jointly (L-BFGS, coarse to fine in the degree, on a
demodulated `crop`² grid); the per-image phases (`arg c_m`) follow in closed form at the optimum, so they include the
distortion. ψ contains only degrees 2…`degree`, orthogonalized (with the data weights) against a constant and a linear term:
those are the phases and the k-vectors, never the distortion.
+ `model = :displacement`: one displacement field D(x, y) of the whole pattern, ψ_i = k_i·D (all orders and directions in
  sync); `:direction`: one phase field per direction (group of parallel peaks), harmonics scaled by their |k|.
+ `reg`: penalty `reg Σ_j deg_j² θ_j²` on the coefficients (radians at the base frequency), relative to the summed peak
  strength at the start.
+ `start_psi`: optionally one more start: per group (`peak_groups`) the local phase of its base peak on the data grid
  (e.g. a previous estimate, `distortion_phase`); fitted like the surrogate's unwrapped phase.
+ `init`: the start of the fit. `:zero` (no distortion) gets trapped in local optima for large distortions (≳ 1 rad RMS);
  `:unwrap` starts from the surrogate solution: per direction the demodulated peak region (the `crop`² grid) of all images,
  phase-aligned and summed, is a field ≈ |·| e^{iΦ(x)}; its phase is unwrapped (quality-guided, strongest pixels first) and
  the polynomial fitted by weighted least squares, then its amplitude scanned (0.5…3×, with the k-vectors re-fitted). The
  images are aligned by their phase relative to the strongest image (the distortion cancels in it). `:auto` (default) runs
  both starts and keeps the higher peak strength
  (`:direction` model: per direction, then refined jointly).
Returns the new `SIMParams` (refined k-vectors and phases, `distortion` set; strengths unchanged) and
`(; gain, rms, start)`: the relative increase of the summed peak strength, the RMS of ψ (radians) per group and the
start that won (`:zero` / `:unwrap` / `:given`, or `:combined`: per direction the better one, `:direction` model).
"""
function estimate_distortion(dat, sp::SIMParams, mypsf=sp.mypsf; degree=3, model=:displacement, reg=1e-4, crop=64,
                             suppress_sigma=0.15, subtract_mean=true, otf_exponent=1.0, otf_moebius=1.0, iterations=300,
                             init::Symbol=:auto, start_psi=nothing, verbose=false)
    init in (:auto, :zero, :unwrap) || throw(ArgumentError("init must be :auto, :zero or :unwrap"))
    model in (:displacement, :direction) || throw(ArgumentError("model must be :displacement or :direction"))
    degree >= 2 || throw(ArgumentError("a distortion needs degree ≥ 2 (degree 0 and 1 are the phases and the k-vectors)"))
    d3 = ndims(dat) == 4 ? dat[:, :, 1, :] : dat
    ndims(d3) == 3 || throw(ArgumentError("expected images (x, y, images) or (x, y, 1, images), got size $(size(dat))"))
    sz = size(d3)[1:2]
    M = size(d3, 3)
    M == size(sp.peak_phases, 1) || throw(ArgumentError("$M images but the SIM parameters describe $(size(sp.peak_phases, 1))"))
    c = 2 * (min(crop, minimum(sz)) ÷ 2)

    # The full-size work (filtering, correlation products, FFTs, the final phases) runs in Float32 on the device of `dat`
    # (a CuArray: the GPU); only the small c×c fit grid and the L-BFGS parameters are on the CPU (Float64).
    todev(x) = copyto!(similar(d3, eltype(x), size(x)), x)
    # the correlation products P_m = (image_m ⊛ psf) · conj(mean ⊛ psf), as `estimate_parameters` / `get_rel_subpixel_correl`
    d3 = Float32.(d3)
    mymean = mean(d3; dims=3)
    _, corr_psf, cropped = precompute_correlations(d3, mypsf; otf_moebius=otf_moebius, otf_exponent=otf_exponent,
                                                   suppress_sigma=suppress_sigma, subtract_mean=subtract_mean, datmean=mymean)
    corr_psf = todev(eltype(corr_psf) <: Complex ? ComplexF32.(Array(corr_psf)) : Float32.(Array(corr_psf)))
    wf = mymean[:, :, 1]
    ref = conj.(FindShift.prepare_correlation(wf, corr_psf; upsample=false))
    P = [FindShift.prepare_correlation(cropped[:, :, m], corr_psf; upsample=false) .* ref for m in 1:M]

    groups = peak_groups(sp)
    G = length(groups)
    G > 0 || error("no pattern peaks: nothing to estimate")
    base = [g[1] for g in groups]
    imgs = [findall(m -> any(sp.peak_strengths[m, i] != 0 for i in g), 1:M) for g in groups]
    kpx = [sp.k_peak_pos[b][1:2] .* sz ./ 2 for b in base]                 # the peaks in Fourier pixels (as `kvecs_to_peak`)
    demod = [todev(collect(exp_ikx_sep(Array{ComplexF32,2}, sz; shift_by=Tuple(k)))) for k in kpx]
    # demodulated by the start k-vector, low-passed and sampled on a c×c grid (Fourier crop): the slowly varying residual
    ctr = sz .÷ 2 .+ 1
    rng = ntuple(d -> ctr[d]-c÷2:ctr[d]+c÷2-1, 2)
    coarse(Q) = ComplexF64.(Array(ifft(ifftshift(fftshift(fft(Q))[rng...]))))     # (small: on the CPU)
    Qc = [[coarse(P[m] .* demod[g]) for m in imgs[g]] for g in 1:G]

    # the basis: Legendre products of degree 2…degree, orthogonalized against 1, u, v with the data weights
    U = repeat(_norm_axis(c), 1, c); V = repeat(_norm_axis(c)', c, 1)
    w = vec(sum(sum(abs.(q) for q in Qg) for Qg in Qc))
    terms = distortion_terms(degree)
    tid = findall(t -> sum(t) >= 2, terms)
    Lu, Lv = _legendre(degree, _norm_axis(c)), _legendre(degree, _norm_axis(c))
    L3 = hcat(ones(c * c), vec(U), vec(V))
    Gram = L3' * (w .* L3)
    nb = length(tid)
    B = zeros(c * c, nb)
    T = zeros(length(terms), nb)            # θ → full Legendre coefficients (incl. the removed constant and linear parts)
    for (j, t) in enumerate(tid)
        a, b = terms[t]
        bt = vec(Lu[a+1, :] * Lv[b+1, :]')
        β = Gram \ (L3' * (w .* bt))
        B[:, j] .= bt .- L3 * β
        T[t, j] = 1.0
        T[1:3, j] .-= β                     # terms 1:3 are (0,0), (1,0), (0,1)
    end
    degj = [sum(terms[t]) for t in tid]

    # parameters: per group the linear phase (α, β) (the k-vector refinement), then θ (:direction: nb per group;
    # :displacement: nb for Dx, nb for Dy, in radians at the mean base frequency)
    kref = sum(hypot(sp.k_peak_pos[b][1:2]...) for b in base) / G
    kdir = [sp.k_peak_pos[b][1:2] ./ kref for b in base]
    nθ = model == :direction ? G * nb : 2nb
    np = 2G + nθ
    θblock(p, g) = model == :direction ? view(p, 2G + (g-1)*nb + 1:2G + g*nb) : nothing
    function phase!(Φ, p, g)
        Φ .= p[2g-1] .* vec(U) .+ p[2g] .* vec(V)
        if model == :direction
            Φ .+= B * θblock(p, g)
        else
            Φ .+= B * (kdir[g][1] .* view(p, 2G+1:2G+nb) .+ kdir[g][2] .* view(p, 2G+nb+1:2G+2nb))
        end
        return Φ
    end
    Fn = sum(sum(abs2(sum(abs, q)) for q in Qg) for Qg in Qc)
    regs = 0.0                              # (reg relative to the peak strength at the start, set below)
    Φ = zeros(c * c)
    function fg!(F, Gr, p)
        Gr === nothing || fill!(Gr, 0.0)
        val = 0.0
        for g in 1:G
            phase!(Φ, p, g)
            e = cis.(.-Φ)
            W = zeros(ComplexF64, c * c)
            for q in Qc[g]
                qe = vec(q) .* e
                cm = sum(qe)
                val += abs2(cm)
                Gr === nothing || (W .+= conj(cm) .* qe)
            end
            if Gr !== nothing
                dΦ = 2 .* imag.(W) ./ Fn                  # ∂(Σ|c|²/Fn)/∂Φ(x)
                Gr[2g-1] -= dot(dΦ, vec(U)); Gr[2g] -= dot(dΦ, vec(V))
                Bd = B' * dΦ
                if model == :direction
                    Gr[2G + (g-1)*nb + 1:2G + g*nb] .-= Bd
                else
                    Gr[2G+1:2G+nb] .-= kdir[g][1] .* Bd
                    Gr[2G+nb+1:2G+2nb] .-= kdir[g][2] .* Bd
                end
            end
        end
        θ = view(p, 2G+1:np)
        dg = repeat(degj, nθ ÷ nb)
        pen = regs * sum(dg .^ 2 .* θ .^ 2)
        Gr === nothing || (Gr[2G+1:np] .+= 2regs .* dg .^ 2 .* θ)
        F === nothing || return -val / Fn + pen
        return nothing
    end
    start = -fg!(0.0, nothing, zeros(np))
    regs = reg * start
    # coarse to fine from `p` (first the k-vectors alone, then the distortion up to degree 2, 3, …); returns p and the loss
    function refine(p; dmaxes=1:degree)
        p = copy(p)
        for dmax in dmaxes
            act = vcat(trues(2G), dmax >= 2 ? repeat(degj .<= dmax, nθ ÷ nb) : falses(nθ))
            x0 = p[act]
            full = copy(p)
            function fga!(F, Gr, x)
                full[act] .= x
                Gf = Gr === nothing ? nothing : zeros(np)
                r = fg!(F, Gf, full)
                Gr === nothing || (Gr .= Gf[act])
                return r
            end
            res = Optim.optimize(x -> fga!(0.0, nothing, x), (Gr, x) -> fga!(nothing, Gr, x), x0, Optim.LBFGS(),
                                 Optim.Options(iterations=iterations, g_tol=1e-10))
            p[act] .= Optim.minimizer(res)
        end
        return p, fg!(0.0, nothing, p)
    end
    # Start values: zero (the undistorted estimate), and the surrogate solution — the local phase map of the demodulated
    # peak region (the c×c grid), unwrapped and fitted by least squares (`_unwrap_start`; no local optima for a smooth
    # field). Large distortions (≳ 1 rad RMS) trap the fit from zero in a local optimum; `init` = :zero, :unwrap or :auto
    # (both, the better one is kept).
    starts = Any[(init == :auto ? [:zero, :unwrap] : [init])...]
    start_psi === nothing || push!(starts, :given)
    # per group: its summed peak strength and penalty (the directions are independent in the :direction model)
    function group_loss(p, g)
        Φg = zeros(c * c)
        phase!(Φg, p, g)
        e = cis.(.-Φg)
        v = sum(abs2(sum(vec(q) .* e)) for q in Qc[g]) / Fn
        θ = model == :direction ? θblock(p, g) : view(p, 2G+1:np)
        return -v + regs * sum(repeat(degj, length(θ) ÷ nb) .^ 2 .* θ .^ 2)
    end
    # The surrogate start has the right shape but not reliably the right amplitude (unwrapping errors in gaps of a sparse
    # object can exaggerate or dilute it): scan a factor on its distortion (with the k-vectors re-fitted for each), per
    # direction (:direction model) or for all, and keep the best before the full refinement.
    function scale_scan(p0)
        facs = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
        cands = Any[]
        for f in facs
            pf = copy(p0); pf[2G+1:np] .*= f
            push!(cands, first(refine(pf; dmaxes=1:1)))
        end
        if model == :direction
            pb = copy(cands[1])
            for g in 1:G
                c_ = cands[argmin([group_loss(cd, g) for cd in cands])]
                pb[2g-1:2g] .= c_[2g-1:2g]
                pb[2G + (g-1)*nb + 1:2G + g*nb] .= c_[2G + (g-1)*nb + 1:2G + g*nb]
            end
            return pb
        end
        return cands[argmin([fg!(0.0, nothing, cd) for cd in cands])]
    end
    fits = Any[]
    for st in starts
        p0 = st == :zero ? zeros(np) :
             st == :given ? _unwrap_start(Qc, U, V, B, kdir, model, G, nb;
                                          Φs=[vec(Float64.(ψ[1:size(ψ, 1) ÷ c:end, 1:size(ψ, 2) ÷ c:end][1:c, 1:c])) for ψ in start_psi]) :
             _unwrap_start(Qc, U, V, B, kdir, model, G, nb)
        st == :unwrap && (p0 = scale_scan(p0))
        pf, L = refine(p0)
        verbose && println("distortion fit from $st start: peak strength $(round(-L; sigdigits=5)) (undistorted $(round(start; sigdigits=5)))")
        push!(fits, (p=pf, L=L, st=st))
    end
    best = fits[argmin([f.L for f in fits])]
    if model == :direction && length(fits) > 1
        # the better start for every direction, combined and refined once more
        pc = zeros(np)
        for g in 1:G
            f = fits[argmin([group_loss(f.p, g) for f in fits])]
            pc[2g-1:2g] .= f.p[2g-1:2g]
            pc[2G + (g-1)*nb + 1:2G + g*nb] .= f.p[2G + (g-1)*nb + 1:2G + g*nb]
        end
        pr, Lr = refine(pc)
        Lr < best.L && (best = (p=pr, L=Lr, st=:combined))
    end
    p = best.p
    final = -fg!(0.0, nothing, p) + regs * sum(repeat(degj, nθ ÷ nb) .^ 2 .* p[2G+1:np] .^ 2)

    # ψ of every peak (Q convention = the phase convention of the correlation: the pattern's ψ, see the tests)
    coefs = zeros(length(sp.k_peak_pos), length(terms))
    ψcoef(g) = model == :direction ? T * θblock(p, g) :
                                     T * (kdir[g][1] .* p[2G+1:2G+nb] .+ kdir[g][2] .* p[2G+nb+1:2G+2nb])
    rms = zeros(G)
    for (g, gr) in enumerate(groups)
        cg = ψcoef(g)
        rms[g] = sqrt(sum(abs2, _eval_legendre(cg, degree, (c, c))) / c^2)
        for i in gr
            r = hypot(sp.k_peak_pos[i][1:2]...) / hypot(sp.k_peak_pos[gr[1]][1:2]...)
            coefs[i, :] .= r .* cg
        end
    end
    # the refined k-vectors: the linear phase α u + β v over the full grid is a shift of the peak by (α, β)/π Fourier pixels
    newk = copy(sp.k_peak_pos)
    for (g, gr) in enumerate(groups)
        dk = _DISTORTION_KSIGN .* (p[2g-1], p[2g]) ./ π
        for i in gr
            r = hypot(sp.k_peak_pos[i][1:2]...) / hypot(sp.k_peak_pos[gr[1]][1:2]...)
            k = sp.k_peak_pos[i]
            newk[i] = (k[1] + r * 2 * dk[1] / sz[1], k[2] + r * 2 * dk[2] / sz[2], k[3])
        end
    end
    sp2 = SIMParams(sp; k_peak_pos=newk, distortion=PatternDistortion(degree, coefs, model))
    # the per-image phases at the optimum: arg c_m on the full grid with the fitted k and ψ (as `get_rel_subpixel_correl`)
    phases = copy(sp.peak_phases)
    psf2 = FourierTools.conv_psf(corr_psf, corr_psf)
    for (g, gr) in enumerate(groups)
        b = gr[1]
        e = todev(collect(exp_ikx_sep(Array{ComplexF32,2}, sz; shift_by=Tuple(newk[b][1:2] .* sz ./ 2))))
        den = sum(wf .* conj.(FourierTools.conv_psf(wf, psf2 .* e)))       # (the normalization of `get_rel_subpixel_correl`)
        ψ = distortion_phase(sp2, b, sz)
        ψ === nothing || (e .*= todev(cis.(.-Float32.(ψ))))
        for m in imgs[g]
            φ = angle(sum(P[m] .* e) / den)
            for i in gr
                r = hypot(sp.k_peak_pos[i][1:2]...) / hypot(sp.k_peak_pos[b][1:2]...)
                phases[m, i] = sp.peak_phases[m, i] + r * (φ - sp.peak_phases[m, b])
            end
        end
    end
    sp2.peak_phases = phases
    return sp2, (; gain=final / max(start, eps()) - 1, rms, start=best.st)
end

# Quality-guided phase unwrapping of a complex field: from its strongest pixel, always the strongest pixel next to the
# unwrapped region is unwrapped relative to its strongest unwrapped neighbour.
function _unwrap_quality(Z)
    n1, n2 = size(Z)
    q = abs.(Z)
    out = zeros(n1, n2)
    done = falses(n1, n2)
    front = fill(-Inf, n1, n2)               # the quality of the frontier pixels (-Inf: not on the frontier)
    i0 = argmax(q)
    out[i0] = angle(Z[i0]); done[i0] = true
    nbrs(i, j) = ((i + a, j + b) for (a, b) in ((1, 0), (-1, 0), (0, 1), (0, -1)) if 1 <= i + a <= n1 && 1 <= j + b <= n2)
    for (a, b) in nbrs(Tuple(i0)...); done[a, b] || (front[a, b] = q[a, b]); end
    for _ in 2:n1*n2
        v, ci = findmax(front)
        v == -Inf && break
        i, j = Tuple(ci)
        # the strongest unwrapped neighbour
        bi, bj, bq = 0, 0, -Inf
        for (a, b) in nbrs(i, j)
            done[a, b] && q[a, b] > bq && ((bi, bj, bq) = (a, b, q[a, b]))
        end
        out[i, j] = out[bi, bj] + angle(Z[i, j] * conj(Z[bi, bj]))
        done[i, j] = true; front[i, j] = -Inf
        for (a, b) in nbrs(i, j); done[a, b] || (front[a, b] = q[a, b]); end
    end
    return out
end

# The surrogate start of `estimate_distortion`: per group the demodulated peak fields of its images, phase-aligned by
# their global correlation and summed (and slightly smoothed), have the phase Φ(x) = const + α u + β v + ψ(x); unwrapped and
# fitted by weighted least squares (weights |Z|²) in the fit's parameterization (per group α, β; θ of the model).
# Per group the demodulated peak fields of its images, phase-aligned to the strongest image and summed, slightly smoothed:
# ≈ |·| e^{iΦ(x)}, Φ = const + α u + β v + ψ. (The relative phase Σ Q_m conj(Q_ref) does not depend on the distortion — it
# cancels — unlike the global correlation Σ Q_m, which large distortions make ill-defined.)
function _surrogate_fields(Qc, c)
    f = [min(i - 1, c - i + 1) for i in 1:c]
    smooth = exp.(-(f .^ 2 .+ (f') .^ 2) ./ (2 * (c / 8)^2))       # (a Gaussian low-pass, σ ≈ 1.3 grid pixels)
    return map(Qc) do Qg
        ref = Qg[argmax([sum(abs2, q) for q in Qg])]
        Z = zeros(ComplexF64, c, c)
        for q in Qg
            r = sum(q .* conj.(ref))
            Z .+= q .* (abs(r) > 0 ? conj(r) / abs(r) : 1.0)
        end
        ifft(fft(Z) .* smooth)
    end
end

function _unwrap_start(Qc, U, V, B, kdir, model, G, nb; Φs=nothing)
    c = size(U, 1)
    given = Φs !== nothing
    given || (Φs = Vector{Vector{Float64}}(undef, G))
    Ws = Vector{Vector{Float64}}(undef, G)
    Zs = _surrogate_fields(Qc, c)
    for g in 1:G
        Z = Zs[g]
        given || (Φs[g] = vec(_unwrap_quality(Z)))
        Ws[g] = vec(abs2.(Z))
    end
    nθ = model == :direction ? G * nb : 2nb
    np = 2G + nθ
    # columns: per group a constant (not a parameter), then the parameters
    A = zeros(G * c * c, G + np)
    y = zeros(G * c * c); w = zeros(G * c * c)
    for g in 1:G
        r = (g - 1) * c * c + 1:g * c * c
        A[r, g] .= 1
        A[r, G + 2g - 1] .= vec(U); A[r, G + 2g] .= vec(V)
        if model == :direction
            A[r, G + 2G + (g - 1) * nb + 1:G + 2G + g * nb] .= B
        else
            A[r, G + 2G + 1:G + 2G + nb] .= kdir[g][1] .* B
            A[r, G + 2G + nb + 1:G + 2G + 2nb] .= kdir[g][2] .* B
        end
        y[r] .= Φs[g]; w[r] .= Ws[g] ./ max(sum(Ws[g]), eps())
    end
    sw = sqrt.(w)
    x = (sw .* A) \ (sw .* y)
    return x[G+1:end]
end

# the sign relating the fitted linear phase to the k-vector shift (`exp_ikx_sep` convention, checked in the tests)
const _DISTORTION_KSIGN = 1
