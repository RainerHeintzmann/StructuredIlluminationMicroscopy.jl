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
    lattice_peaks(k_vecs; coherent=false, zero_beam=0, layout=k_vecs)

Peaks of a lattice illumination built from the lateral first-order k-vectors
`k_vecs` (one per direction, relative to Nyquist).

+ `coherent=false`: incoherent sum of the gratings `1 + cos(π k_d⋅x)`: the zero
  order (strength = number of directions) and one peak per direction (strength 1).
+ `coherent=true`: all beams at `±k_d/2` interfere coherently (each grating is
  the interference of such a beam pair). Besides the zero order and the `k_d`
  this contains the mixed peaks `(±k_i ± k_j)/2`; the strength of each peak is
  the number of beam pairs producing it.

`zero_beam > 0`: the zero order of the grating is not blocked. A central beam of
amplitude `zero_beam` (relative to the other beams) is added and the other beams
are at `±k_d` instead, so that `k_d` is the first order (central beam with one
of the `±k_d` beams) as for classical three-beam SIM. This adds the second
orders `2 k_d` and, for `coherent=true`, the mixed peaks `±k_i ± k_j`. For
`coherent=false` every grating is such a three-beam pattern.

The strength of a peak is the sum of the amplitude products of the beam pairs
producing it (the number of pairs, for equal amplitudes).

Returns `(k_peak_pos, strengths)`: the zero order first, then the `k_d` in the
given order, then any mixed or higher peaks. Only one of `±K` is listed.

`layout`: k-vectors (e.g. the ideal ones) that decide which beam pairs produce
the same peak, while the positions are taken from `k_vecs`. Peaks that coincide
exactly for the ideal vectors (e.g. `k_1 + k_3 = k_2` for three directions at
60° with a zero-order beam) then stay one peak for slightly different measured
or pixel-rounded `k_vecs`, which could not be unmixed as separate peaks anyway.
"""
function lattice_peaks(k_vecs; coherent::Bool=false, zero_beam::Real=0, layout=k_vecs)
    kv = [(Float64(k[1]), Float64(k[2])) for k in k_vecs]
    lv0 = [(Float64(k[1]), Float64(k[2])) for k in layout]
    length(lv0) == length(kv) || throw(ArgumentError("layout needs one k-vector per direction ($(length(kv)))."))
    # pair each k-vector with the most parallel unused layout vector, with its sign
    # (k-vectors may be given in another order or as -k)
    lv = similar(lv0)
    used = falses(length(lv0))
    for (i, k) in enumerate(kv)
        cosine(l) = (k[1] * l[1] + k[2] * l[2]) / max(hypot(k...) * hypot(l...), 1e-300)
        j = argmax(j -> used[j] ? -Inf : abs(cosine(lv0[j])), eachindex(lv0))
        used[j] = true
        lv[i] = cosine(lv0[j]) < 0 ? (-lv0[j][1], -lv0[j][2]) : lv0[j]
    end
    counts = Dict{NTuple{2,Float64},Float64}()
    rep = Dict{NTuple{2,Float64},NTuple{2,Float64}}()   # canonical key (of the layout) -> vector as listed
    order = NTuple{2,Float64}[]
    add!(K, L, c) = begin
        key = _round_k(_canonical_k(L))
        if !haskey(counts, key)
            counts[key] = 0.0
            rep[key] = K
            push!(order, key)
        end
        counts[key] += c
    end
    for (k, l) in zip(kv, lv)   # list the first orders first, with the given sign
        add!(k, l, 0.0)
    end
    # Beams (position, layout position, amplitude) of each mutually coherent
    # group: all beams (coherent) or the beams of each grating (sum of
    # gratings). Zero order blocked: ±k_d/2; otherwise the central beam and ±k_d.
    h = zero_beam == 0 ? 0.5 : 1.0
    z = (0.0, 0.0)
    center = zero_beam == 0 ? Tuple{NTuple{2,Float64},NTuple{2,Float64},Float64}[] : [(z, z, Float64(zero_beam))]
    groups = coherent ? [vcat(center, [(k .* h, l .* h, 1.0) for (k, l) in zip(kv, lv)],
                                      [(k .* -h, l .* -h, 1.0) for (k, l) in zip(kv, lv)])] :
                        [vcat(center, [(k .* h, l .* h, 1.0), (k .* -h, l .* -h, 1.0)]) for (k, l) in zip(kv, lv)]
    zero = 0.0
    for g in groups, (qa, la, A) in g, (qb, lb, B) in g
        L = la .- lb
        if hypot(L...) < 1e-9
            zero += A * B
        else
            add!(qa .- qb, L, A * B)   # (a,b) and (b,a) both land on the same ±K key
        end
    end
    # Intensity = Σ_ab A_a A_b exp(i (q_a - q_b)⋅x): the zero order counts the
    # self-pairs, a ±K pair contributes 2 c_K cos(K⋅x); normalized by 1/2 so
    # that a single beam pair gives the grating 1 + cos(K⋅x).
    zero_strength = zero / 2
    strengths_of = key -> counts[key] / 2
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
    generate_lattice_peaks(k_vecs; coherent=false, zero_beam=0, layout=k_vecs, scan=:line, num_images=0, raster=(3, 3), shifts=nothing)

Like `generate_peaks`, but for a lattice illumination (see `lattice_peaks` and
`lattice_shifts`). Pass `shifts` (one (x, y) translation in raw pixels per image)
to use given translations instead of the optimized ones.

Returns `k_peak_pos, peak_phases, peak_strengths, otf_indices, otf_phases`.
"""
function generate_lattice_peaks(k_vecs; coherent::Bool=false, zero_beam::Real=0, layout=k_vecs, scan::Symbol=:line,
                                num_images::Integer=0, raster=(3, 3), shifts=nothing)
    k_peak_pos, strengths = lattice_peaks(k_vecs; coherent=coherent, zero_beam=zero_beam, layout=layout)
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
