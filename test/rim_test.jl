using Test, StructuredIlluminationMicroscopy, Statistics, Random
using FFTW: fft, ifft, ifftshift, fftshift, fftfreq
using StructuredIlluminationMicroscopy.IndexFunArrays: idx

# RIM (second-order statistics) reconstruction of nine-frame SIM data
@testset "RIM" begin
    nx, df, kc, k1 = 64, 2, 0.5, 0.45
    q = [hypot(fftfreq(nx, nx)[i], fftfreq(nx, nx)[j]) / (nx / 2 * kc) for i in 1:nx, j in 1:nx]
    psf = fftshift(real.(ifft([x >= 1 ? 0.0 : (2 / π) * (acos(x) - x * sqrt(1 - x^2)) for x in q]))); psf ./= sum(psf)
    mypsf = reshape(psf, nx, nx, 1)
    sp = generate_peaks_param(mypsf, 9, 3, 0.0, 2, k1, true, 0.0)

    # pattern statistics: equal phase steps are second-order stationary, g_E = Σ (1/6) cos(k·d) per direction
    kv, w, res, μE = rim_pattern_statistics(sp)
    @test length(kv) == 3 && all(isapprox.(w, 1 / 6; atol=1e-12)) && res < 1e-12 && μE ≈ 1
    @test all(any(k -> isapprox(collect(k), collect(sp.k_peak_pos[i][1:2]); atol=1e-12) ||
                       isapprox(collect(k), -collect(sp.k_peak_pos[i][1:2]); atol=1e-12), kv) for i in 2:length(sp.k_peak_pos))
    spu = generate_peaks_param(mypsf, 9, 3, 0.0, 2, k1, true, 0.0)
    spu.peak_phases[[1, 4, 7], :] .= 0; spu.peak_phases[[2, 5, 8], :] .*= 0.3        # uneven phase steps: not stationary
    @test rim_pattern_statistics(spu)[3] > 0.1

    # the adjoint of the variance model (finite differences)
    let n = 16, pos = idx(Float64, (16, 16))
        H = fft(ifftshift(exp.(-(getindex.(pos, 1) .^ 2 .+ getindex.(pos, 2) .^ 2) ./ 4)))
        ramps = [cis.(0.7 .* getindex.(pos, 1)), ones(ComplexF64, n, n)]
        wt = [0.3, 0.1]
        Random.seed!(3); ρ = rand(n, n); G = randn(n, n)
        f(ρ) = sum(G .* StructuredIlluminationMicroscopy._rim_forward(ρ, H, ramps, wt)[1])
        _, A = StructuredIlluminationMicroscopy._rim_forward(ρ, H, ramps, wt)
        g = StructuredIlluminationMicroscopy._rim_adjoint(G, A, H, ramps, wt)
        e = zeros(n, n); e[37] = 1e-6
        @test isapprox(g[37], (f(ρ .+ e) - f(ρ .- e)) / 2e-6; rtol=1e-5)
    end

    # reconstruction of noiseless data: better than widefield, aligned with the object, non-negative
    Random.seed!(5)
    obj = zeros(Float32, nx * df, nx * df, 1)
    for _ in 1:150; obj[rand(1:nx*df), rand(1:nx*df), 1] = 1 + rand(); end
    data, _ = simulate_sim_3d(obj, sp, mypsf; n_photons=0.0, downsample_factor=df)
    o = Float64.(obj[:, :, 1])
    ρ, info = rim_reconstruct(Float64.(data), sp; iterations=150, noise_gain=0.0)
    @test size(ρ) == size(o) && all(ρ .>= 0)
    wf = StructuredIlluminationMicroscopy.resample_by_rft(Float64.(dropdims(sum(data; dims=4); dims=(3, 4))), size(o))
    @test cor(vec(ρ), vec(o)) > cor(vec(wf), vec(o)) + 0.05
    @test info.loss[end] < 0.5 * info.loss[1]
    @test_throws ArgumentError rim_reconstruct(Float64.(data[:, :, 1, 1:8]), sp)        # images and parameters disagree

    # the leading term of the variance kernel t(x,x') = h(x) h(x') Σ w cos(k·(x-x')) is an eigenvector: T u₁ = λ u₁
    let pos = idx(Float64, (32, 32)), h = exp.(-(getindex.(pos, 1) .^ 2 .+ getindex.(pos, 2) .^ 2) ./ 8)
        ramps = [cis.(0.6 .* getindex.(pos, 1)), cis.(0.3 .* getindex.(pos, 1) .+ 0.5 .* getindex.(pos, 2))]
        wt = [0.2, 0.1]
        u = StructuredIlluminationMicroscopy._rim_leading_kernel(h, ramps, wt)
        Tu = h .* sum(wt[j] .* real.(ramps[j] .* sum(conj.(ramps[j]) .* h .* u)) for j in 1:2)
        @test isapprox(Tu, (sum(Tu .* u) / sum(abs2, u)) .* u; rtol=1e-8) && sum(u) > 0
    end
    # variance fit with the RIM-CF start
    ρs, infos = rim_reconstruct(Float64.(data), sp; match=:var, init=:cf, iterations=60, noise_gain=0.0)
    @test all(ρs .>= 0) && infos.loss[end] < 0.5 * infos.loss[1]
    @test cor(vec(ρs), vec(o)) > cor(vec(wf), vec(o))
end
