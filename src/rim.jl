"""
Random illumination microscopy (RIM) reconstruction of SIM data,
after Pseudo-Random Illumination Microscopy (P-RIM), Barbault, Idier, Allain, Mangeat, Labouesse, Sentenac,
Inverse Problems 42, 025006 (2026), hal-04977967.

RIM processes a set of structured-illumination images by matching SECOND-ORDER statistics, as random illumination
microscopy (RIM) does, instead of demodulating every image with its known pattern as SIM does. With the empirical
moments over the `M` images `I_m`,

    V(r) = 1/M Σ_m (I_m(r) − μ_I(r))² = ∬ ρ(x) ρ(x') h(r−x) h(r−x') g_E(x−x') dx dx',

which holds when the pattern set is second-order stationary (SOS: the empirical mean and covariance of the patterns
are shift invariant, e.g. the nine-frame 2-D SIM set or the 7-frame hexagonal lattice). The covariance is then harmonic,
`g_E(d) = Σ_j w_j cos(k_j·d)`, so only the pattern FREQUENCIES and their weights are needed, not the phase of each image,
and `V(r) = Σ_j w_j |(h ⊛ (ρ e^{i k_j·x}))(r)|²`. The identifiable support is the same as for SIM.

Here the variance (or standard deviation, "RIM-STD") image is fitted by a non-negative object on a finer grid
(preconditioned non-linear conjugate gradients, Giroussens et al., IEEE TCI 2024). 2-D data only.
"""

"""
    rim_pattern_statistics(sp::SIMParams; tol=1e-6)

The second-order statistics of the illumination patterns of `sp` (one 2-D pattern per image, see `SIMPattern`).
Returns `(kvecs, weights, sos_residual, mean_level)`:
+ `kvecs`, `weights`: the empirical pattern covariance is `g_E(d) = Σ_j weights[j] cos(π kvecs[j]·d)` (`kvecs` relative to the
  Nyquist frequency of the images as `sp.k_peak_pos`, `d` in pixels; one entry per ±k pair, `k = 0` if the zero order varies).
+ `sos_residual`: the relative size of the non-stationary part of the covariance (0 for an SOS set). Sets with a large residual
  (e.g. phase steps that do not cover the circle evenly) violate the RIM model.
+ `mean_level`: the mean pattern intensity (zero-order coefficient averaged over the images).
Axial components of the peaks are ignored.
"""
function rim_pattern_statistics(sp::SIMParams; tol=1e-6)
    M = size(sp.peak_phases, 1)
    ks = NTuple{2,Float64}[]
    cs = Vector{ComplexF64}[]
    function add!(k, c)
        j = findfirst(q -> abs(q[1] - k[1]) < tol && abs(q[2] - k[2]) < tol, ks)
        if isnothing(j)
            push!(ks, k); push!(cs, copy(c))
        else
            cs[j] .+= c
        end
    end
    for i in eachindex(sp.k_peak_pos)
        k = (Float64(sp.k_peak_pos[i][1]), Float64(sp.k_peak_pos[i][2]))
        s = Float64.(sp.peak_strengths[:, i])
        φ = Float64.(sp.peak_phases[:, i])
        if abs(k[1]) < tol && abs(k[2]) < tol
            add!((0.0, 0.0), complex.(s))
        else                                     # s cos(k·x + φ) = s/2 e^{iφ} e^{ik·x} + c.c.
            add!(k, s ./ 2 .* cis.(φ))
            add!(.-k, s ./ 2 .* cis.(.-φ))
        end
    end
    n = length(ks)
    mc = [sum(c) / M for c in cs]
    C = [sum(cs[a] .* conj.(cs[b])) / M - mc[a] * conj(mc[b]) for a in 1:n, b in 1:n]   # cov of the coefficients
    diagC = real.([C[a, a] for a in 1:n])
    offd = sum(abs2(C[a, b]) for a in 1:n, b in 1:n if a != b; init=0.0)
    sos_residual = sqrt(offd) / max(sqrt(sum(abs2, diagC)), eps())
    kvecs = NTuple{2,Float64}[]
    weights = Float64[]
    for a in 1:n
        diagC[a] > tol * maximum(diagC; init=0.0) || continue
        k = ks[a]
        iszero_k = abs(k[1]) < tol && abs(k[2]) < tol
        # keep one of each ±k pair: the pair contributes 2 C_aa cos(k·d)
        if !iszero_k && (k[1] < -tol || (abs(k[1]) < tol && k[2] < 0))
            continue
        end
        push!(kvecs, k)
        push!(weights, iszero_k ? diagC[a] : 2 * diagC[a])
    end
    j0 = findfirst(q -> abs(q[1]) < tol && abs(q[2]) < tol, ks)
    mean_level = isnothing(j0) ? 0.0 : real(mc[j0])
    return kvecs, weights, sos_residual, mean_level
end

# Fourier interpolation of a 2-D image by an integer factor, values preserved (the pixel (1,1) stays at (1,1)).
function _rim_upsample(img::AbstractMatrix, u::Int)
    u == 1 && return Float64.(img)
    return resample_by_rft(Float64.(img), size(img) .* u) .* u^2
end

# In-place FFT plans of the complex work array `C`, and the filtering of `C` by the Fourier multiplier `K` in place.
_rim_plans(C) = (plan_fft!(C), plan_ifft!(C))
_rim_filter!(C, K, pf, pinv) = (pf * C; C .*= K; pinv * C; C)

# The variance model V = Σ_j w_j |A_j|², A_j = h ⊛ (ρ e_j), into `V` and `A` (kept for the gradient); `C` is the work array.
function _rim_forward!(V, A, ρ, H, ramps, weights, C, pf, pinv)
    fill!(V, 0)
    for j in eachindex(ramps)
        C .= ρ .* ramps[j]
        _rim_filter!(C, H, pf, pinv)
        A[j] .= C
        V .+= weights[j] .* abs2.(C)
    end
    return V
end

# Gradient of Σ_r G(r) V(r) with respect to ρ, into `g` (`Hc = conj.(H)`).
function _rim_adjoint!(g, G, A, Hc, ramps, weights, C, pf, pinv)
    fill!(g, 0)
    for j in eachindex(ramps)
        C .= G .* conj.(A[j])
        _rim_filter!(C, Hc, pf, pinv)
        g .+= (2 * weights[j]) .* real.(ramps[j] .* C)
    end
    return g
end

# (allocating versions)
function _rim_forward(ρ, H, ramps, weights)
    C = similar(ρ, complex(eltype(ρ)))
    A = [similar(C) for _ in ramps]
    V = _rim_forward!(similar(ρ), A, ρ, H, ramps, weights, C, _rim_plans(C)...)
    return V, A
end
function _rim_adjoint(G, A, H, ramps, weights)
    C = similar(G, complex(eltype(G)))
    return _rim_adjoint!(similar(G), G, A, conj.(H), ramps, weights, C, _rim_plans(C)...)
end

# The Fourier symbol of the quadratic form Σ_r V(r) = ρᵀSρ: s̃(ν) = Σ_j w_j |H(ν + k_j)|² (symmetrised, ρ is real), the sum of
# the shifted squared OTFs as in the SIM Wiener denominator. Used as preconditioner (Giroussens et al., IEEE TCI 2024).
function _rim_otf_sum(hf, ramps, weights)
    s = zeros(Float64, size(hf))
    for j in eachindex(ramps)
        s .+= weights[j] / 2 .* (abs2.(fft(ifftshift(hf .* ramps[j]))) .+ abs2.(fft(ifftshift(hf .* conj.(ramps[j])))))
    end
    return s
end

# The leading term u₁ of the spectral decomposition of the variance kernel t(x, x') = h(x) h(x') g_E(x − x'), so that
# V ≈ (u₁ ⊛ ρ)² (RIM-CF, Giroussens et al., IEEE TCI 2024). For the harmonic pattern covariance the kernel has the finite
# rank 2·length(kvecs) with the basis h cos(k·x), h sin(k·x); `hf` and `ramps` are centred. Returns u₁ (centred, sum > 0).
function _rim_leading_kernel(hf, ramps, weights)
    B = Matrix{Float64}(undef, length(hf), 0)
    wb = Float64[]
    for j in eachindex(ramps)
        B = hcat(B, vec(hf .* real.(ramps[j])), vec(hf .* imag.(ramps[j])))
        append!(wb, (weights[j], weights[j]))
    end
    keep = [norm(B[:, c]) > 1e-12 * norm(hf) for c in axes(B, 2)]      # (sin of the zero frequency)
    B = B[:, keep]; wb = wb[keep]
    sw = sqrt.(max.(wb, 0.0))
    F = eigen(Symmetric((B' * B) .* (sw * sw')))
    u = reshape(B * (sw .* F.vectors[:, end]), size(hf))
    return sum(u) < 0 ? -u : u
end

"""
    rim_reconstruct(dat, sp::SIMParams; upsample_factor=2, match=:std, offset=0.0, background=0.0, noise_gain=1.0, read_noise_var=0.0,
                     mean_weight=1.0, reg=1e-3, iterations=100, init=:mean, plane=nothing, tol=1e-7, verbose=false)

RIM-STD (or RIM-variance) reconstruction of 2-D structured-illumination data `dat` (`(nx, ny, M)`, or `(nx, ny, nz, M)` of
which the plane `plane` (default: the centre) is used) with the pattern frequencies and strengths of `sp` (the phases only
enter through the pattern covariance, `rim_pattern_statistics`) and the PSF `sp.mypsf` (the data size; the centre plane of a
3-D PSF). Returns `(ρ, info)` with the object `ρ ≥ 0` on a grid `upsample_factor` times finer (pixel (1,1) of the data at
(1,1)) and `info = (loss, iterations, sos_residual, kvecs, weights)`.

+ `match`: `:std` fits the standard deviation `sqrt(V)` (as RIM-STD; less dominated by bright regions, and its Hessian is
  close to the preconditioner), `:var` the variance. With the noise variance in the standard-deviation model `:std` was better
  in all simulations (nine-frame SIM, with and without photon noise and pattern distortions).
+ `offset` (camera offset) and `background` (a constant signal, e.g. out-of-focus light) are subtracted from the data.
+ Noise: the variance of the photon noise biases the empirical variance; `noise_gain` (variance per count, 1 for data in photons,
  0 to switch off) times the mean signal plus `background` (not the offset), plus `read_noise_var`. For `:var` it is subtracted
  from the data, for `:std` it is added to the model, `sqrt(V(ρ) + σ²)` (as RIM-STD). The target is limited to the support of
  the model (twice the PSF bandwidth), which removes the noise outside it.
+ `mean_weight`: relative weight of also matching the mean image `μ_I = μ_E h ⊛ ρ` (0: second order only, as in the paper).
  The mean is not affected by pattern distortions and much less noisy than the variance; in simulations (nine-frame SIM,
  sparse object) it raised the NCC with the object from 0.40 to 0.63 at 1000 photons and from 0.52 to 0.79 without noise.
+ `reg`: weight of a smoothness penalty Σ|∇ρ|² (relative to the data term). The optimizer converges fast, so it fits the
  noise unless `reg` is raised for noisy data (simulations: 1e-4 best for sparse noiseless objects, 1e-2 for filaments at 1000
  photons).
+ `dat` on the GPU (a `CuArray`): the iterations run there, in the element type of `dat` (Float32 recommended), and `ρ` is
  returned there; the one-time preparation (upsampling, statistics, preconditioner) runs on the CPU.
+ `init`: start value, `:cf` the closed-form RIM-CF estimate (the standard deviation image deconvolved with the leading term of
  the variance kernel), `:mean` the deconvolved mean image, `:flat` a constant.
The optimizer follows RIM-STD (Giroussens et al., "Fast super-resolved reconstructions in fluorescence random illumination
microscopy", IEEE Trans. Comput. Imaging 2024): non-linear conjugate gradients preconditioned with the shifted-OTF sum
(diagonal in Fourier space), a Newton step along the direction (exact minimum of the quartic for `:var`), Armijo backtracking,
and ρ ≥ 0 (projection). The constraint is essential: the variance does not change when ρ changes sign locally, and without it
the fit stalls in such sign-flipped solutions (the paper fits without it, on smooth objects).
"""
function rim_reconstruct(dat, sp::SIMParams; upsample_factor::Int=2, match::Symbol=:std, offset=0.0, background=0.0, noise_gain=1.0,
                          read_noise_var=0.0, mean_weight=1.0, reg=1e-3, iterations::Int=100, init::Symbol=:mean,
                          plane=nothing, tol=1e-7, verbose=false)
    match in (:std, :var) || throw(ArgumentError("match must be :std or :var"))
    init in (:cf, :mean, :flat) || throw(ArgumentError("init must be :cf, :mean or :flat"))
    frames = let d = Array(dat)
        if ndims(d) == 4
            p = isnothing(plane) ? size(d, 3) ÷ 2 + 1 : plane
            Float64.(d[:, :, p, :])
        elseif ndims(d) == 3
            Float64.(d)
        else
            throw(ArgumentError("RIM needs 2-D images stacked along the last dimension, got size $(size(d))"))
        end
    end
    nx, ny, M = size(frames)
    M == size(sp.peak_phases, 1) || throw(ArgumentError("$M images but the SIM parameters describe $(size(sp.peak_phases, 1))"))
    kvecs, weights, sos_residual, mean_level = rim_pattern_statistics(sp)
    isempty(kvecs) && error("the illumination patterns do not vary: RIM needs structured illumination")
    sos_residual > 0.05 && @warn "the pattern set is not second-order stationary (relative residual $(round(sos_residual; digits=3))): the RIM model is approximate"

    u = upsample_factor
    fsz = (nx * u, ny * u)
    up = zeros(Float64, fsz..., M)
    for m in 1:M
        up[:, :, m] .= _rim_upsample(frames[:, :, m] .- (offset + background), u)
    end
    μ = dropdims(sum(up; dims=3); dims=3) ./ M
    Vemp = dropdims(sum(abs2, up .- μ; dims=3); dims=3) ./ M
    # noise variance (its bias in the empirical (1/M) variance: (M-1)/M σ², σ² = gain·(signal + background) + read noise)
    Vnoise = (M - 1) / M .* (noise_gain .* max.(μ .+ background, 0) .+ read_noise_var)

    psf2 = let p = sp.mypsf
        ndims(p) > 2 && size(p, 3) > 1 ? p[:, :, size(p, 3) ÷ 2 + 1] : p[:, :]
    end
    hf = _rim_upsample(Float64.(psf2), u)
    hf ./= sum(hf)
    H = fft(ifftshift(hf))
    # support of the model: FT(h²) = H ⋆ H is non-zero up to twice the PSF bandwidth
    Wsup = abs.(fft(ifftshift(hf .^ 2)))
    Wsup = Float64.(Wsup .> 1e-4 * maximum(Wsup))
    band(x) = real.(ifft(Wsup .* fft(x)))
    # :var subtracts the noise variance from the data; :std keeps it in the model, sqrt(V(ρ) + Vnoise), and fits the square
    # root of the raw empirical variance (as RIM-STD), which avoids the bias of the clipped sqrt(max(Vemp - Vnoise, 0))
    target, nmodel = if match == :std
        sqrt.(max.(band(Vemp), 0.0)), max.(Vnoise, 0.0) .+ 1e-9 * maximum(abs, Vemp)
    else
        band(Vemp .- Vnoise), nothing
    end
    tnorm = max(sum(abs2, target), eps())

    pos = idx(Float64, fsz)
    ramps = [cis.((π / u) .* (k[1] .* getindex.(pos, 1) .+ k[2] .* getindex.(pos, 2))) for k in kvecs]

    # the mean image is μ_E (h ⊛ ρ) (start value and the optional mean term)
    μnorm = max(sum(abs2, μ), eps())

    ρ = if init == :cf
        # RIM-CF: the standard deviation ≈ u₁ ⊛ ρ, Wiener-deconvolved with the leading kernel term
        U1 = fft(ifftshift(_rim_leading_kernel(hf, ramps, weights)))
        σt = sqrt.(max.(band(Vemp .- Vnoise), 0.0))
        max.(real.(ifft(conj.(U1) .* fft(σt) ./ (abs2.(U1) .+ 1e-2 * maximum(abs2, U1)))), 0.0)
    elseif init == :mean && abs(mean_level) > eps()
        # Wiener deconvolution of the mean image
        max.(real.(ifft(conj.(H) .* fft(μ) ./ (abs2.(H) .+ 1e-2))) ./ mean_level, 0.0)
    else
        ones(Float64, fsz)
    end
    sum(ρ) > 0 || (ρ = ones(Float64, fsz))
    let (V0, _) = _rim_forward(ρ, H, ramps, weights)       # scaled to the variance
        Vm = match == :std ? sqrt.(V0 .+ nmodel) : V0
        c = sum(target .* Vm) / max(sum(abs2, Vm), eps())
        ρ .*= match == :std ? c : sqrt(max(c, 0.0))
    end
    ρscale2 = max(sum(abs2, ρ), eps())

    kx = 2π .* fftfreq(fsz[1]); ky = 2π .* fftfreq(fsz[2])
    lap = [2 * (1 - cos(a)) + 2 * (1 - cos(b)) for a in kx, b in ky]    # Fourier symbol of -∇² (finite differences)

    # Preconditioner (Giroussens et al.): a Gauss-Newton approximation of the Hessian that is diagonal in Fourier space, the
    # shifted-OTF sum s̃ for the data term (scaled by its local gain: ∂V/∂ρ ∝ sqrt(V) for :var, ∂sqrt(V+n)/∂ρ ≈ const for :std),
    # |H|² for the mean term, the Laplacian for the smoothness term
    P = let (V0, _) = _rim_forward(ρ, H, ramps, weights)
        c = match == :std ? 0.5 * sum(V0) / max(sum(V0 .+ nmodel), eps()) : 2 * sum(V0) / length(V0)
        p = (c / tnorm) .* _rim_otf_sum(hf, ramps, weights)
        mean_weight > 0 && (p .+= (mean_weight * mean_level^2 / μnorm) .* abs2.(H))
        reg > 0 && (p .+= (reg / ρscale2) .* lap)
        p .+ 1e-6 * maximum(p)
    end

    # to the array type of the data (the GPU for a CuArray) and its element type; scalars too, so that nothing is promoted
    T = dat isa Array ? Float64 : float(real(eltype(dat)))
    todev(x::AbstractArray{<:Complex}) = copyto!(similar(dat, Complex{T}, size(x)), x)
    todev(x::AbstractArray) = copyto!(similar(dat, T, size(x)), x)
    todev(::Nothing) = nothing
    H, target, nmodel, μ, lap, P, ρ0 = todev.((H, target, nmodel, μ, lap, P, ρ))
    Hc = conj.(H)
    ramps = todev.(ramps)
    weights = T.(weights)
    tnorm, μnorm, ρscale2, mean_level, mean_weight, reg = T.((tnorm, μnorm, ρscale2, mean_level, mean_weight, reg))

    # All work arrays are allocated here and the FFTs run in place in `C`: the iterations allocate no arrays. (On the GPU the
    # GC frees nothing while the caller holds a lock, e.g. the GUI's reconstruction lock: temporaries filled the memory.)
    C = similar(H)
    pf, pinv = _rim_plans(C)
    rbuf() = similar(ρ0)
    # the state at the current and at the trial point (swapped on acceptance): ρ, V, A_j, the gradient, the mean residual, -∇²ρ
    newstate() = (ρ=rbuf(), V=rbuf(), A=[similar(C) for _ in ramps], g=rbuf(), Rm=rbuf(), Lρ=rbuf())
    cur, tr = newstate(), newstate()
    copyto!(cur.ρ, ρ0)
    G, W1, W2, V1, V2, d, gF, gF_prev, z = (rbuf() for _ in 1:9)
    filt!(out, x, K) = (C .= x; _rim_filter!(C, K, pf, pinv); out .= real.(C))

    function loss_grad!(st)
        local L       # (not the optimizer's variable of the same name: a closure would assign it)
        _rim_forward!(st.V, st.A, st.ρ, H, ramps, weights, C, pf, pinv)
        L = 0.0
        if match == :std
            W1 .= sqrt.(st.V .+ nmodel)
            W2 .= W1 .- target
            L += 0.5 * sum(abs2, W2) / tnorm
            G .= W2 ./ (2 .* W1) ./ tnorm
        else
            W2 .= st.V .- target
            L += 0.5 * sum(abs2, W2) / tnorm
            G .= W2 ./ tnorm
        end
        _rim_adjoint!(st.g, G, st.A, Hc, ramps, weights, C, pf, pinv)
        if mean_weight > 0
            filt!(st.Rm, st.ρ, H)
            st.Rm .= mean_level .* st.Rm .- μ
            L += 0.5 * mean_weight * sum(abs2, st.Rm) / μnorm
            filt!(W1, st.Rm, Hc)
            st.g .+= (mean_weight * mean_level / μnorm) .* W1
        end
        if reg > 0
            filt!(st.Lρ, st.ρ, lap)
            L += 0.5 * reg * dot(st.ρ, st.Lρ) / ρscale2
            st.g .+= (reg / ρscale2) .* st.Lρ
        end
        return L
    end

    # The first and second derivative of the loss along d at α = 0 (and, for :var, the cubic of its exact derivative):
    # with D_j = h ⊛ (d e_j), V(ρ + αd) = V + α V1 + α² V2, V1 = 2 Σ w Re(conj(A_j) D_j), V2 = Σ w |D_j|²
    function line_step(st, αprev)
        local α
        fill!(V1, 0); fill!(V2, 0)
        for j in eachindex(ramps)
            C .= d .* ramps[j]
            _rim_filter!(C, H, pf, pinv)
            V1 .+= (2 * weights[j]) .* real.(conj.(st.A[j]) .* C)
            V2 .+= weights[j] .* abs2.(C)
        end
        a0 = 0.0; a1 = 0.0                 # φ'(α) ≈ a0 + a1 α (+ a2 α² + a3 α³ for :var)
        if mean_weight > 0
            filt!(W1, d, H)                # (the mean model changes by mean_level·W1)
            a0 += mean_weight * mean_level * dot(st.Rm, W1) / μnorm
            a1 += mean_weight * mean_level^2 * sum(abs2, W1) / μnorm
        end
        if reg > 0
            filt!(W1, d, lap)
            a0 += reg * dot(d, st.Lρ) / ρscale2
            a1 += reg * dot(d, W1) / ρscale2
        end
        if match == :std
            W2 .= sqrt.(st.V .+ nmodel)                                  # S
            W1 .= V1 ./ (2 .* W2)                                        # S' = V1 / 2S
            V2 .= V2 ./ W2 .- V1 .^ 2 ./ (4 .* W2 .^ 3)                  # S'' = V2 / S - V1² / 4S³
            W2 .-= target                                                # S - target
            a0 += dot(W2, W1) / tnorm
            a1 += (sum(abs2, W1) + dot(W2, V2)) / tnorm
            return a1 > 0 ? -a0 / a1 : αprev
        else
            W2 .= st.V .- target
            a0 += dot(W2, V1) / tnorm
            a1 += (sum(abs2, V1) + 2 * dot(W2, V2)) / tnorm
            a2 = 3 * dot(V1, V2) / tnorm
            a3 = 2 * sum(abs2, V2) / tnorm
            α = a1 > 0 ? -a0 / a1 : αprev   # Newton on the cubic φ'(α), started at the quadratic approximation
            for _ in 1:30
                dφ = a0 + α * (a1 + α * (a2 + α * a3)); d2φ = a1 + α * (2a2 + 3a3 * α)
                d2φ > 0 || break
                δ = dφ / d2φ; α -= δ
                abs(δ) <= 1e-10 * abs(α) && break
            end
            return α > 0 ? α : αprev
        end
    end

    # Preconditioned non-linear conjugate gradients (Polak-Ribière+), projected on ρ ≥ 0, with the Newton step along the
    # direction as first trial and Armijo backtracking. Pixels at the bound that the gradient pushes below it (the active
    # set) are taken out before preconditioning, otherwise the preconditioned direction need not descend.
    function free_dir!(st)
        gF .= ifelse.((st.ρ .<= 0) .& (st.g .> 0), zero(T), st.g)
        C .= gF; pf * C; C ./= P; pinv * C
        z .= ifelse.((st.ρ .<= 0) .& (st.g .> 0), zero(T), real.(C))
        return nothing
    end
    L = loss_grad!(cur)
    losses = [L]
    free_dir!(cur)
    d .= .-z
    gz_prev = dot(gF, z); gF_prev .= gF
    α = 1.0
    it = 0
    for outer it in 1:iterations
        local Ln
        d .= ifelse.((cur.ρ .<= 0) .& ((cur.g .> 0) .| (d .< 0)), zero(T), d)   # (removing these keeps g·d < 0: there g ≤ 0)
        if dot(cur.g, d) >= 0                 # not a descent direction: restart with the preconditioned gradient
            d .= ifelse.((cur.ρ .<= 0) .& (z .> 0), zero(T), .-z)
            dot(cur.g, d) < 0 || break
        end
        α = line_step(cur, α)
        accepted = false
        for _ in 1:30
            tr.ρ .= max.(cur.ρ .+ T(α) .* d, zero(T))
            Ln = loss_grad!(tr)
            W1 .= tr.ρ .- cur.ρ
            if Ln <= L + 1e-4 * dot(cur.g, W1)
                accepted = true
                break
            end
            α /= 2
        end
        accepted || break
        rel = abs(L - Ln) / max(abs(L), eps())
        cur, tr = tr, cur
        L = Ln
        push!(losses, L)
        free_dir!(cur)
        W1 .= gF .- gF_prev
        β = T(max(dot(W1, z) / max(gz_prev, eps()), 0.0))
        d .= β .* d .- z
        gz_prev = dot(gF, z); gF_prev .= gF
        verbose && println("RIM iteration $it: loss $(L), step $α, β $(round(β; sigdigits=3))")
        it > 5 && rel < tol && break
    end
    return cur.ρ, (loss=losses, iterations=it, sos_residual=sos_residual, kvecs=kvecs, weights=weights)
end
