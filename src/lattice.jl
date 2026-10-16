# Two-dimensional ("lattice") SIM illumination: every raw image contains the
# modulation of all directions at once, and the images differ by translations
# of the whole pattern. The orders are then unmixed by the same pseudo-inverse
# of the weight matrix as for classical SIM (see `pinv_weight_matrix`).
#
# Conventions (as in `SIMPattern`): k-vectors are relative to the Nyquist
# frequency of the raw images, so a translation of the pattern by `s` raw
# pixels changes the phase of the peak at `k` by `-π k⋅s`. A peak strength `a`
# of a non-zero peak means a modulation `a⋅cos(π k⋅x + φ)`.

"""
    ideal_lattice_kvecs(num_directions, k1, phi0=0.0)

First-order lateral k-vectors (relative to Nyquist) of `num_directions`
directions at the angles `π d / num_directions + phi0`, as used by `generate_peaks`.
"""
ideal_lattice_kvecs(num_directions::Integer, k1, phi0=0.0) =
    [(k1 * cos(pi * d / num_directions + phi0), k1 * sin(pi * d / num_directions + phi0)) for d in 0:num_directions-1]

# Canonical representative of ±K (both describe the same real modulation).
function _canonical_k(K::NTuple{2,Float64}; tol=1e-9)
    (K[1] > tol || (abs(K[1]) <= tol && K[2] > tol)) && return K
    return (-K[1], -K[2])
end
_round_k(K::NTuple{2,Float64}) = (round(K[1]; digits=9) + 0.0, round(K[2]; digits=9) + 0.0)

"""
    lattice_peaks(k_vecs; coherent=false)

Peaks of a lattice illumination built from the lateral first-order k-vectors
`k_vecs` (one per direction, relative to Nyquist).

+ `coherent=false`: incoherent sum of the gratings `1 + cos(π k_d⋅x)`: the zero
  order (strength = number of directions) and one peak per direction (strength 1).
+ `coherent=true`: all beams at `±k_d/2` interfere coherently (each grating is
  the interference of such a beam pair). Besides the zero order and the `k_d`
  this contains the mixed peaks `(±k_i ± k_j)/2`; the strength of each peak is
  the number of beam pairs producing it.

Returns `(k_peak_pos, strengths)`: the zero order first, then the `k_d` in the
given order, then any mixed peaks. Only one of `±K` is listed.
"""
function lattice_peaks(k_vecs; coherent::Bool=false)
    kv = [(Float64(k[1]), Float64(k[2])) for k in k_vecs]
    counts = Dict{NTuple{2,Float64},Float64}()
    rep = Dict{NTuple{2,Float64},NTuple{2,Float64}}()   # canonical key -> vector as listed
    order = NTuple{2,Float64}[]
    add!(K, c) = begin
        key = _round_k(_canonical_k(K))
        if !haskey(counts, key)
            counts[key] = 0.0
            rep[key] = K
            push!(order, key)
        end
        counts[key] += c
    end
    for k in kv               # list the first orders first, with the given sign
        add!(k, 0.0)
    end
    if coherent
        beams = vcat([k ./ 2 for k in kv], [k ./ -2 for k in kv])
        zero = 0.0
        for a in beams, b in beams
            K = a .- b
            if hypot(K...) < 1e-9
                zero += 1
            else
                add!(K, 1.0)   # (a,b) and (b,a) both land on the same ±K key
            end
        end
        # Intensity = Σ_ab exp(i (q_a - q_b)⋅x): the zero order counts the
        # self-pairs, a ±K pair contributes 2 c_K cos(K⋅x); normalized by 1/2 so
        # that a single beam pair gives the grating 1 + cos(K⋅x).
        zero_strength = zero / 2
        strengths_of = key -> counts[key] / 2
    else
        for k in kv
            add!(k, 1.0)
        end
        zero_strength = Float64(length(kv))
        strengths_of = key -> counts[key]
    end
    keys_used = [key for key in order if counts[key] > 0]
    k_peak_pos = NTuple{3,Float64}[(0.0, 0.0, 0.0)]
    strengths = Float64[zero_strength]
    for key in keys_used
        K = rep[key]
        push!(k_peak_pos, (K[1], K[2], 0.0))
        push!(strengths, strengths_of(key))
    end
    return k_peak_pos, strengths
end

# Weight matrix (images × peaks) for the given translations, as `weight_matrix`.
function _lattice_weights(k_peak_pos, strengths, shifts)
    phases = [-pi * (k[1] * s[1] + k[2] * s[2]) for s in shifts, k in k_peak_pos]
    strength_mat = repeat(reshape(strengths, 1, :), length(shifts), 1)
    return phases, strength_mat
end

# Conditioning quality (1/cond, 1 = best) of unmixing all peaks and their
# conjugates from the images taken at `shifts` (cf. `pinv_weight_matrix`).
function _lattice_quality(k_peak_pos, strengths, shifts)
    phases, s = _lattice_weights(k_peak_pos, strengths, shifts)
    wm = cis.(phases) .* s
    ext = hcat(wm, conj.(wm[:, 2:end]))
    c = cond(ext)
    return isfinite(c) ? 1 / c : 0.0
end

"""
    lattice_shifts(k_peak_pos, strengths; scan=:line, num_images=0, raster=(3, 3), steps=121)

Translations (in raw pixels) of the lattice pattern for each image, chosen for
the best conditioning of the unmixing:

+ `scan=:line`: `num_images` translations `p⋅s` along a single step vector `s`.
+ `scan=:raster`: an XY raster of `raster[1]` × `raster[2]` translations
  `(i Δx, j Δy)` (x fastest).

The step(s) are found by a grid search (`steps` values per coordinate) within
one period of the coarsest modulation. `num_images` must be at least the number
of real unknowns, `2 * length(k_peak_pos) - 1`.
"""
function lattice_shifts(k_peak_pos, strengths; scan::Symbol=:line, num_images::Integer=0,
                        raster=(3, 3), steps::Integer=121)
    knorms = [hypot(k[1], k[2]) for k in k_peak_pos[2:end]]
    filter!(>(1e-9), knorms)
    isempty(knorms) && throw(ArgumentError("A lattice needs at least one non-zero k-vector."))
    period = 2 / minimum(knorms)            # raw pixels per period of the coarsest peak
    make(sx, sy) = scan == :raster ?
        [(i * sx, j * sy) for j in 0:raster[2]-1 for i in 0:raster[1]-1] :
        [(p * sx, p * sy) for p in 0:num_images-1]
    scan == :raster || num_images > 0 || error("num_images is required for scan=:line")
    rng = range(scan == :raster ? period / steps : -period, period; length=steps)
    best = (-1.0, 0.0, 0.0)
    for sx in rng, sy in rng
        q = _lattice_quality(k_peak_pos, strengths, make(sx, sy))
        q > best[1] && (best = (q, sx, sy))
    end
    return make(best[2], best[3]), best[1]
end

"""
    generate_lattice_peaks(k_vecs; coherent=false, scan=:line, num_images=0, raster=(3, 3), shifts=nothing)

Like `generate_peaks`, but for a lattice illumination (see `lattice_peaks` and
`lattice_shifts`). Pass `shifts` (one (x, y) translation in raw pixels per image)
to use given translations instead of the optimized ones.

Returns `k_peak_pos, peak_phases, peak_strengths, otf_indices, otf_phases`.
"""
function generate_lattice_peaks(k_vecs; coherent::Bool=false, scan::Symbol=:line, num_images::Integer=0,
                                raster=(3, 3), shifts=nothing)
    k_peak_pos, strengths = lattice_peaks(k_vecs; coherent=coherent)
    if isnothing(shifts)
        shifts, quality = lattice_shifts(k_peak_pos, strengths; scan=scan, num_images=num_images, raster=raster)
    else
        quality = _lattice_quality(k_peak_pos, strengths, shifts)
    end
    nmin = 2 * length(k_peak_pos) - 1
    length(shifts) < nmin &&
        @warn "Lattice SIM with $(length(k_peak_pos)) peaks needs at least $nmin images to unmix all orders, got $(length(shifts))."
    quality < 1e-3 && @warn "Lattice translations unmix the orders poorly (conditioning quality $(round(quality; sigdigits=3)))."
    peak_phases, peak_strengths = _lattice_weights(k_peak_pos, strengths, shifts)
    otf_indices, otf_phases = make_3d_pattern(k_peak_pos, 0.0)
    return k_peak_pos, peak_phases, peak_strengths, otf_indices, otf_phases
end

"""
    generate_lattice_param(mypsf, k_vecs; kwargs...)

`generate_lattice_peaks` packaged into a `SIMParams` object.
"""
function generate_lattice_param(mypsf, k_vecs; kwargs...)
    k_peak_pos, peak_phases, peak_strengths, otf_indices, otf_phases = generate_lattice_peaks(k_vecs; kwargs...)
    return SIMParams(mypsf, k_peak_pos, peak_phases, peak_strengths, otf_indices, otf_phases)
end
