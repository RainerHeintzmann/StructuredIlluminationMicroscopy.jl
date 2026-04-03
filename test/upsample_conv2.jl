using Test
using StructuredIlluminationMicroscopy
using Tullio
using OffsetArrays

@testset "upsample_conv2! basics" begin
    # Identity kernels should behave like repeat(src, inner=(2,2))
    src = zeros(Float64, 5, 5)
    src[3,3] = 1.0
    kI = [0.0, 1.0, 0.0]
    dst = zeros(Float64, 10, 10)
    upsample_conv2!(dst, src, kI, kI, kI, kI)
    @test dst == repeat(src, inner=(2,2))
end

@testset "upsample_conv2! sum scaling" begin
    src = reshape(1:9, 3, 3) .|> float
    k = [0.25, 0.5, 0.25]
    dst = zeros(Float64, 6, 6)
    upsample_conv2!(dst, src, k, k, k, k)
    # With kx_even=kx_odd=k and ky_even=ky_odd=k, total sum scales by 4*sum(k)^2
    @test isapprox(sum(dst), 4 * sum(src) * sum(k) * sum(k); atol=1e-10)
    @test size(dst) == (2*size(src,1), 2*size(src,2))
end
