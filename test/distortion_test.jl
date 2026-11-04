using Test, StructuredIlluminationMicroscopy, Statistics, Random, LinearAlgebra
using FFTW: fft, ifft, ifftshift, fftshift, fftfreq

# smooth illumination distortions of the simulation (pattern_displacement, SIMPattern's displacement)
@testset "pattern distortion" begin
    nx = 64
    q = [hypot(fftfreq(nx, nx)[i], fftfreq(nx, nx)[j]) / (nx / 2 * 0.5) for i in 1:nx, j in 1:nx]
    psf = fftshift(real.(ifft([x >= 1 ? 0.0 : (2 / π) * (acos(x) - x * sqrt(1 - x^2)) for x in q]))); psf ./= sum(psf)
    sp = generate_peaks_param(reshape(psf, nx, nx, 1), 9, 3, 0.0, 2, 0.45, true, 0.0)
    Random.seed!(2)
    dc = pattern_displacement((nx, nx, 1); relative=0.5, scale=1.0, shape=:cos)
    @test size(dc[1]) == (nx, nx) && maximum(hypot.(dc...)) <= 0.5 + 1e-9 && maximum(hypot.(dc...)) > 0.45
    dg = pattern_displacement((nx, nx); relative=0.25, scale=0.5, shape=:gaussian)
    @test size(dg[2]) == (nx, nx) && maximum(hypot.(dg...)) > 0
    @test_throws ArgumentError pattern_displacement((nx, nx); shape=:square)
    h = Float64.(psf)
    p0 = StructuredIlluminationMicroscopy.SIMPattern(h, sp, 2, 1)
    @test StructuredIlluminationMicroscopy.SIMPattern(h, sp, 2, 1; displacement=(zeros(nx, nx), zeros(nx, nx))) ≈ p0
    pd = StructuredIlluminationMicroscopy.SIMPattern(h, sp, 2, 1; displacement=dc)
    @test !(pd ≈ p0)                                          # the fringes move,
    @test isapprox(sum(pd), sum(p0); rtol=0.05)               # the mean stays
end

# estimation (estimate_distortion: k-vectors, distortion and phases jointly) and compensation in the reconstruction
@testset "distortion estimation" begin
    S = StructuredIlluminationMicroscopy
    nx, df, kc, k1 = 96, 2, 0.5, 0.45
    q = [hypot(fftfreq(nx, nx)[i], fftfreq(nx, nx)[j]) / (nx / 2 * kc) for i in 1:nx, j in 1:nx]
    psf = fftshift(real.(ifft([x >= 1 ? 0.0 : (2 / π) * (acos(x) - x * sqrt(1 - x^2)) for x in q]))); psf ./= sum(psf)
    mypsf = reshape(psf, nx, nx, 1)
    sp = generate_peaks_param(mypsf, 9, 3, 0.0, 2, k1, true, 0.0)
    N = nx * df
    Random.seed!(1)
    obj = zeros(Float32, N, N, 1)
    for _ in 1:40
        x0, y0, a = rand() * N, rand() * N, rand() * π
        for t in 0:0.5:50
            obj[mod1(round(Int, x0 + t * cos(a)), N), mod1(round(Int, y0 + t * sin(a)), N), 1] += 1
        end
    end
    groups = peak_groups(sp)
    @test length(groups) == 3 && all(length.(groups) .== 1)
    seeds = [Tuple(sp.k_peak_pos[g[1]][1:2] .* nx ./ 2) for g in groups]
    est(dat) = estimate_parameters(dat, mypsf; k_vecs=seeds, num_directions=3, show_quality=false, verbose=false)
    # the true local phase of peak i (pattern convention) on the data grid
    function true_psi(D, i)
        Dx, Dy = D isa Tuple ? D : D[S._peak_group_of(sp, i)]
        dscale = 2 / minimum(hypot(p[1:2]...) for p in sp.k_peak_pos if hypot(p[1:2]...) > 0) * df
        k = π .* sp.k_peak_pos[i][1:2] ./ df
        return (dscale .* (k[1] .* Dx .+ k[2] .* Dy))[1:df:end, 1:df:end]
    end
    function detrend(ψ)
        u = S._norm_axis(size(ψ, 1)); v = S._norm_axis(size(ψ, 2))
        A = hcat(ones(length(ψ)), vec(repeat(u, 1, size(ψ, 2))), vec(repeat(v', size(ψ, 1), 1)))
        return vec(ψ) .- A * (A \ vec(ψ))
    end
    @test length(pattern_displacement((N, N); relative=0.2, per_direction=3)) == 3

    # undistorted data: (almost) no distortion, k-vectors and phases as estimate_parameters
    data0, _ = simulate_sim_3d(obj, sp, mypsf; n_photons=0.0, downsample_factor=df)
    d0 = Float32.(data0[:, :, 1, :])
    spe0 = est(d0)
    spd0, info0 = estimate_distortion(d0, spe0, mypsf; degree=3)
    # (a sparse object correlates a little with any smooth phase: small spurious ψ)
    @test all(info0.rms .< 0.2) && info0.gain < 0.04
    @test all(maximum(abs.(spd0.k_peak_pos[i][1:2] .- sp.k_peak_pos[i][1:2]) .* nx ./ 2) < 0.1 for i in 2:4)
    @test maximum(abs.(angle.(cis.(spd0.peak_phases .- spe0.peak_phases)))) < 0.3
    @test_throws ArgumentError estimate_distortion(d0, spe0, mypsf; degree=1)
    @test_throws ArgumentError estimate_distortion(d0, spe0, mypsf; model=:other)
    # a wrong start k (0.4 Fourier pixels off) is pulled back: the sign of the k-vector update
    spw = SIMParams(spe0); k = spw.k_peak_pos[2]; spw.k_peak_pos[2] = (k[1] + 0.8 / nx, k[2], k[3])
    spw2, _ = estimate_distortion(d0, spw, mypsf; degree=2)
    @test abs(spw2.k_peak_pos[2][1] - spe0.k_peak_pos[2][1]) * nx / 2 < 0.15
    # distortion-free parameters keep the separable reconstruction path
    @test isempty(S.distortion_phasors(d0, spe0)) && distortion_phase(spe0, 2, (nx, nx)) === nothing

    # out-of-sync distortion (each direction displaced by its own field): ψ found, reconstruction improved
    Random.seed!(3)
    D = pattern_displacement((N, N); relative=0.5, scale=1.0, shape=:gaussian, per_direction=3)
    data, _ = simulate_sim_3d(obj, sp, mypsf; n_photons=0.0, downsample_factor=df, displacement=D)
    d = Float32.(data[:, :, 1, :])
    spe = est(d)
    spd, info = estimate_distortion(d, spe, mypsf; degree=4, model=:direction)
    @test spd.distortion isa PatternDistortion && spd.distortion.model == :direction && info.gain > 0.05
    for g in groups
        @test cor(detrend(distortion_phase(spd, g[1], (nx, nx))), detrend(true_psi(D, g[1]))) > 0.8
    end
    rp = ReconParams(; double_use=false, upsample_factor=df)
    # the super-resolved band of the reconstruction (same order masks for both): better with the distortion
    fr = [hypot(fftfreq(N, N)[i], fftfreq(N, N)[j]) for i in 1:N, j in 1:N]
    hp(a) = real.(ifft(fft(Float64.(a)) .* ((fr .> kc * nx / 2) .& (fr .< (kc + k1) * nx / 2))))
    prepD = recon_sim_prepare(data, spd, rp; use_final_filter=true)
    rD = recon_sim(data, prepD, spd)
    prepS = recon_sim_prepare(data, spe, rp; use_final_filter=true); prepS.distortion_masks = prepD.distortion_masks
    rS = recon_sim(data, prepS, spe)
    o = Float64.(obj[:, :, 1])
    @test cor(vec(hp(rD[:, :, 1])), vec(hp(o))) > cor(vec(hp(rS[:, :, 1])), vec(hp(o))) + 0.03
    # in-sync model (one displacement of the whole pattern)
    Random.seed!(4)
    Ds = pattern_displacement((N, N); relative=0.4, scale=1.0, shape=:gaussian)
    datas, _ = simulate_sim_3d(obj, sp, mypsf; n_photons=0.0, downsample_factor=df, displacement=Ds)
    ds = Float32.(datas[:, :, 1, :])
    sps, _ = estimate_distortion(ds, est(ds), mypsf; degree=4, model=:displacement)
    for g in groups
        @test cor(detrend(distortion_phase(sps, g[1], (nx, nx))), detrend(true_psi(Ds, g[1]))) > 0.6
    end
    # sampling-independent: resampling keeps the distortion; SIMPattern applies it
    @test S.resample_sim_params(spd, 2).distortion === spd.distortion
    h = Float64.(psf)
    @test !(S.SIMPattern(h, spd, 2, 1) ≈ S.SIMPattern(h, SIMParams(spd; distortion=nothing), 2, 1))
end
