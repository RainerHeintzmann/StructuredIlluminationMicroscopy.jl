using Test, StructuredIlluminationMicroscopy, Statistics, Random
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
