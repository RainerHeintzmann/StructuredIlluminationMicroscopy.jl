using Test, StructuredIlluminationMicroscopy, Statistics, Random
using FFTW: fft, ifft, ifftshift, fftshift, fftfreq

# A scalar wide-field 3-D PSF (angular spectrum of a circular pupil), centred, sum 1.
function widefield_psf3d(nx, nz; dx=0.115, dz=0.15, NA=1.1, n=1.33, λ=0.58)
    f = fftfreq(nx, 1 / dx)
    k2 = [f[i]^2 + f[j]^2 for i in 1:nx, j in 1:nx]
    pupil = k2 .<= (NA / λ)^2
    kz = sqrt.(max.((n / λ)^2 .- k2, 0.0))
    h = zeros(nx, nx, nz)
    for (iz, z) in enumerate(((0:nz-1) .- nz ÷ 2) .* dz)
        h[:, :, iz] = abs2.(ifft(pupil .* cis.(2π .* kz .* z)))
    end
    h = fftshift(h, (1, 2))
    return h ./ sum(h)
end

# 3-D SIM (three beams, second order at the Abbe limit, λ_ex = λ_em): the OTF-matched 3-D estimate locates the
# k-vectors and phases of a thick sample; the z-average estimate does not.
@testset "estimate_parameters_3d" begin
    nx, nz, df = 64, 32, 2
    dx, dz, NA, n, λ = 0.115, 0.15, 1.1, 1.33, 0.58
    h = Float32.(widefield_psf3d(nx, nz; dx=dx, dz=dz, NA=NA, n=n, λ=λ))
    k1 = (NA / λ) * 2dx                                   # first order (relative to Nyquist): 2 k1 at 2NA/λ
    k0 = 2 .* (dx, dx, dz) .* n ./ λ
    s = k1 / k0[1]
    kz = (1 - sqrt(1 - s^2)) * k0[3]
    sp = generate_peaks_param(h, 15, 3, 0.1, 3, k1, true, kz)
    Random.seed!(3)
    obj = zeros(Float32, nx * df, nx * df, nz)
    for _ in 1:60                                         # random 3-D filaments through the whole volume
        p = [rand() * nx * df, rand() * nx * df, rand() * nz]
        d = randn(3); d[3] *= 0.3; d ./= sqrt(sum(abs2, d))
        for t in 0:0.5:80
            q = p .+ t .* d
            obj[mod1(round(Int, q[1]), nx * df), mod1(round(Int, q[2]), nx * df), mod1(round(Int, q[3]), nz)] += 1
        end
    end
    dat, _ = simulate_sim_3d(obj, sp, h; n_photons=0, n_photons_bg=0, downsample_factor=df)
    first_orders = [2, 4, 6]
    px(k) = (k[1] * nx / 2, k[2] * nx / 2)
    clicks = [collect(round.(px(sp.k_peak_pos[i]))) for i in first_orders]
    function errors(e)
        ek = [hypot((px(e.k_peak_pos[i]) .- px(sp.k_peak_pos[i]))...) for i in first_orders]
        eph = Float64[]
        for i in first_orders
            idx = findall(sp.peak_strengths[:, i] .!= 0)
            Δ = [angle(cis(e.peak_phases[p, i] - sp.peak_phases[p, i])) for p in idx]
            push!(eph, rad2deg(sqrt(mean(abs2, Δ))))      # absolute: the reconstruction needs the offset too
        end
        return ek, eph
    end
    e3 = estimate_parameters_3d(Float32.(dat), h; k_vecs=clicks, num_directions=3, kz=kz, implied_higher_orders=1)
    @test length(e3.k_peak_pos) == length(sp.k_peak_pos) && size(e3.peak_phases) == size(sp.peak_phases)
    dk, dph = errors(e3)
    @test maximum(dk) < 0.25                              # pixels (|k| ≈ 14 px)
    @test maximum(dph) < 20                               # degrees
    # the implied second orders: twice the first
    @test all(isapprox(collect(e3.k_peak_pos[i + 1][1:2]), 2 .* collect(e3.k_peak_pos[i][1:2]); atol=1e-12) for i in first_orders)
    # the z-average with the 2-D estimate is clearly worse here
    za = dropdims(mean(dat; dims=3); dims=3)
    e2 = estimate_parameters(za, dropdims(sum(h; dims=3); dims=3); k_vecs=clicks, num_directions=3, implied_higher_orders=1,
                             show_quality=false, verbose=false, method=:FindZoomFT)
    dk2, _ = errors(e2)
    @test maximum(dk) < maximum(dk2)
end

# the mean enforcement of preprocess_sim does not depend on the intensity scale
@testset "preprocess_sim scale invariance" begin
    Random.seed!(4)
    dat = 1f-3 .* (1 .+ rand(Float32, 32, 32, 6))
    a = preprocess_sim(dat; bg=0f0, num_directions=2, correct_drift=false, damp_edge=false)
    b = preprocess_sim(1000f0 .* dat; bg=0f0, num_directions=2, correct_drift=false, damp_edge=false)
    @test isapprox(1000f0 .* a, b; rtol=1e-4)
end
