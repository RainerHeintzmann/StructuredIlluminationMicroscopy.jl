using Test
using StructuredIlluminationMicroscopy
using LinearAlgebra: norm
const besseli = StructuredIlluminationMicroscopy.SpecialFunctions.besseli   # (the toolbox depends on SpecialFunctions)

@testset "non-linear illumination schemes" begin
    # saturated depletion only (flat activation, no readout modulation): E = exp(-d (1 - cos v)) = e^-d [I_0(d) + 2 Σ I_m(d) cos(m v)]
    for d in (0.5, 3.0, 12.0)
        s = ActivationDepletionReadout(act_mod=0.0, read_mod=0.0, depl_sat=d, depl_mod=1.0)
        r = scheme_coefficients(s, 6)
        @test r[1] == 1
        @test all(m -> isapprox(r[m + 1], besseli(m, d) / besseli(0, d); rtol=1e-8, atol=1e-12), 0:6)
        str, ph = scheme_strengths(s, 7)
        @test str[1] == 1 && ph[1] == 0
        @test all(m -> isapprox(str[m + 1], 2 * besseli(m, d) / besseli(0, d); rtol=1e-8, atol=1e-8), 1:6)
        @test all(abs.(ph) .< 1e-9)                       # no phase offset without a readout shift
    end
    # stronger saturation → more power in the higher harmonics
    @test scheme_truncation(ActivationDepletionReadout(depl_sat=20.0), 3) > scheme_truncation(ActivationDepletionReadout(depl_sat=1.0), 3)

    # a shifted readout alone: E = 1 + m cos(v - Δ)  →  r_1 = (m/2) e^{-iΔ}, no higher harmonics
    s = ActivationDepletionReadout(act_mod=0.0, depl_sat=0.0, read_mod=0.6, read_shift=0.7)
    r = scheme_coefficients(s, 4)
    @test r[2] ≈ 0.3 * cis(-0.7) atol=1e-10
    @test all(abs.(r[3:end]) .< 1e-10)
    str, ph = scheme_strengths(s, 4)
    @test str ≈ [1.0, 0.6, 0.0, 0.0] atol=1e-9
    @test ph[2] ≈ -0.7 atol=1e-9

    # a readout shifted against the depletion zero gives the harmonics a phase offset (and not m times the first one)
    s = ActivationDepletionReadout(act_mod=0.0, depl_sat=6.0, read_mod=0.9, read_shift=0.8)
    _, ph = scheme_strengths(s, 4)
    @test abs(ph[3] - 2 * ph[2]) > 1e-3 || abs(ph[2]) > 1e-3

    # a custom profile: 1 + cos v is the linear grating
    str, ph = scheme_strengths(PeriodicProfile(v -> 1 + cos(v)), 4)
    @test str ≈ [1.0, 1.0, 0.0, 0.0] atol=1e-9

    # generate_peaks / generate_peaks_directions put the scheme into the SIMParams: the pattern of an image is the profile
    sch = ActivationDepletionReadout(act_sat=0.7, act_mod=0.8, act_shift=0.3, depl_sat=5.0, depl_mod=0.9, read_mod=0.7, read_shift=0.5)
    c0 = real(sum(scheme_profile(sch, 2π * (n - 1) / 4096) for n in 1:4096) / 4096)
    psf = ones(Float32, 32, 32, 1)
    k1, P, no = 0.3, 5, 5
    peaks = Dict("generate_peaks" => generate_peaks_param(psf, 3P, 3, 0.0, no, k1, true, 0.0; scheme=sch),
                 "generate_peaks_directions" => generate_peaks_directions_param(psf, fill(P, 3), fill(no, 3), fill(false, 3), 0.0, k1; scheme=sch))
    for (name, sp) in peaks
        for p in 0:P-1, x in (0.0, 1.7, 5.3, 9.1)
            n = p + 1                                     # an image of the first direction (k along x)
            pattern = sum(sp.peak_strengths[n, i] * (norm(sp.k_peak_pos[i]) == 0 ? 1.0 : cos(π * sp.k_peak_pos[i][1] * x + sp.peak_phases[n, i]))
                          for i in eachindex(sp.k_peak_pos))
            v = π * k1 * x + 2π * p / P
            # truncated to the orders present: compare with the same truncation of the exact profile
            r = scheme_coefficients(sch, no - 1)
            expected = 1 + sum(2 * real(r[m + 1] * cis(m * v)) for m in 1:no-1)
            @test pattern ≈ expected atol=1e-9
        end
        @test sp.peak_strengths[1, 1] == 1                # zero order
    end
    # the untruncated profile is recovered when all harmonics are included (sanity check of the normalization)
    @test scheme_profile(sch, 0.4) / c0 ≈ real(sum(scheme_coefficients(sch, 400)[m + 1] * cis(m * 0.4) * (m == 0 ? 1 : 1) for m in 0:400) +
                                               sum(conj(scheme_coefficients(sch, 400)[m + 1]) * cis(-m * 0.4) for m in 1:400)) atol=1e-6

    # not supported: lattices and blocked zero orders
    @test_throws ErrorException generate_peaks(9, 3, 0.0, 3, 0.3, true, 0.0; use_lattice=true, scheme=sch)
    @test_throws ErrorException generate_peaks_directions([3, 3], [3, 3], [false, true], 0.0, 0.3; scheme=sch)
    # without a scheme nothing changes
    sp0 = generate_peaks_param(psf, 9, 3, 0.0, 3, 0.3, true, 0.0)
    @test all(sp0.peak_strengths[sp0.peak_strengths .!= 0] .== 1)
end

# the model-free estimation of the strengths and phases of the higher orders from simulated non-linear SIM data
@testset "estimate_order_strengths" begin
    using FFTW: fft, ifft, ifftshift, fftshift, fftfreq
    nx, df, nd, M = 128, 3, 2, 3
    P = 2M + 1; no = M + 1; N = nd * P
    kc, k1 = 0.5, 0.45                                   # OTF cutoff and first order (Nyquist units of the raw grid)
    function otf2d(n)
        q = [hypot(fftfreq(n, n)[i], fftfreq(n, n)[j]) / (n / 2 * kc) for i in 1:n, j in 1:n]
        return [x >= 1 ? 0.0 : (2 / π) * (acos(x) - x * sqrt(1 - x^2)) for x in q]
    end
    psf = fftshift(real.(ifft(otf2d(nx)))); psf ./= sum(psf)
    mypsf = reshape(psf, nx, nx, 1)
    obj = zeros(Float32, nx * df, nx * df, 1)
    rng = [(7 * i) % (nx * df) + 1 for i in 1:400]       # a deterministic scatter of points and short lines
    for i in 1:200
        obj[rng[i], rng[(i * 37) % 400 + 1], 1] += 1 + (i % 3)
        for t in 0:40
            a = 0.3 * (i % 7)
            ii = clamp(round(Int, rng[i] + t * cos(a)), 1, nx * df); jj = clamp(round(Int, rng[(i * 11) % 400 + 1] + t * sin(a)), 1, nx * df)
            obj[ii, jj, 1] += 1
        end
    end
    scheme = ActivationDepletionReadout(act_sat=1.0, act_mod=0.7, act_shift=0.4, depl_sat=6.0, depl_mod=0.95, read_mod=0.8, read_shift=0.9)
    str_true, ph_true = scheme_strengths(scheme, no)
    dev_true = [ph_true[m + 1] - m * ph_true[2] for m in 1:M]
    sp_true = generate_peaks_param(mypsf, N, nd, 0.0, no, k1, true, 0.0; scheme=scheme)
    sp_naive = generate_peaks_param(mypsf, N, nd, 0.0, no, k1, true, 0.0)
    for photons in (0.0, 1000.0)
        data, _ = simulate_sim_3d(obj, sp_true, mypsf; n_photons=photons, downsample_factor=df)
        sp_est, info = estimate_order_strengths(Float64.(dropdims(data; dims=3)), sp_naive, mypsf; haze_symmetry=0.15)
        @test length(info) == nd
        for g in info
            @test all(g.valid)
            est = [2 * abs(g.R[m]) for m in 1:M]
            @test all(abs.(est ./ str_true[2:end] .- 1) .< (photons == 0 ? 0.06 : 0.12))      # the strengths, orders beyond the zero-order overlap included
            @test all(abs.(g.phase_dev .- dev_true) .< (photons == 0 ? 0.1 : 0.2))             # phase offsets (deviation from the linear progression)
            @test g.phase_dev[1] == 0
        end
        # the returned SIMParams keep the layout, order 1 keeps its phase, the zero order its strength
        @test size(sp_est.peak_strengths) == size(sp_naive.peak_strengths) && sp_est.k_peak_pos == sp_naive.k_peak_pos
        @test sp_est.peak_phases[:, 2] == sp_naive.peak_phases[:, 2] && sp_est.peak_strengths[:, 1] == sp_naive.peak_strengths[:, 1]
    end
    # estimate_parameters with estimate_orders: the implied higher orders get the measured strengths (the result is normalized to its
    # maximum, so the strengths are compared as ratios); without it they are the ideal ones, and ideal_strength=false no longer
    # broadcasts the first order's magnitude to them
    let
        data, _ = simulate_sim_3d(obj, sp_true, mypsf; n_photons=2000.0, downsample_factor=df)
        dat = Float32.(dropdims(data; dims=3))
        kv = [Tuple(sp_true.k_peak_pos[1 + 1 + (d - 1) * (no - 1)][1:2]) .* nx ./ 2 for d in 1:nd]
        kw = (k_vecs=kv, num_directions=nd, implied_higher_orders=M - 1, refine=false, show_quality=false, verbose=false)
        spi = estimate_parameters(dat, mypsf; kw...)
        @test all(x -> x == 0 || x == spi.peak_strengths[1, 2], spi.peak_strengths[:, 2:end])   # ideal: all present peaks equal (0: peak absent from the image)
        spb = estimate_parameters(dat, mypsf; kw..., ideal_strength=false)
        @test spb.peak_strengths[1, 3] == spb.peak_strengths[1, 4] == spb.peak_strengths[1, 2 + M - 1] && spb.peak_strengths[1, 3] != spb.peak_strengths[1, 2]
        spe = estimate_parameters(dat, mypsf; kw..., ideal_strength=false, estimate_orders=true)
        ratios = spe.peak_strengths[1, 2:no] ./ spe.peak_strengths[1, 2]
        @test all(abs.(ratios ./ (str_true[2:end] ./ str_true[2]) .- 1) .< 0.12)
        @test_throws ErrorException estimate_parameters(dat; kw..., estimate_orders=true)              # needs the PSF
    end
    # too few images for the orders (needs 2M+1 per direction): the unmeasurable orders are absent, not guessed
    data, _ = simulate_sim_3d(obj, sp_true, mypsf; n_photons=0.0, downsample_factor=df)
    sp3 = generate_peaks_param(mypsf, 3nd, nd, 0.0, no, k1, true, 0.0)             # 3 phases per direction only
    _, info3 = estimate_order_strengths(Float64.(dropdims(data[:, :, :, 1:3nd]; dims=3)), sp3, mypsf)
    @test all(g -> count(g.valid) <= 1, info3)
end

@testset "singlet saturation scheme" begin
    lin = scheme_strengths(SingletSaturation(sat=1e-4), 4)[1]          # weak excitation: linear, only the first order
    @test isapprox(lin[2], 1.0; atol=1e-3)
    @test all(lin[3:4] .< 1e-3)
    st = scheme_strengths(SingletSaturation(sat=8.0), 4)[1]            # strong saturation: higher orders appear
    @test st[3] > 0.2 && st[2] > st[3] > st[4]
    @test all(abs.(sin.(scheme_strengths(SingletSaturation(sat=8.0), 4)[2])) .< 1e-9)   # symmetric pattern: phase offsets 0 or π only
end
