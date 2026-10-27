include(joinpath(@__DIR__, "..", "src", "upsample.jl"))

function bench()
    src = rand(Float32, 256, 256)
    base = Float32[0.05, 0.25, 0.4, 0.25, 0.05]  # example base kernel (odd length)
    L = length(base)
    dst_rows = 2*size(src,1) - 2*(L - 1)
    dst_cols = 2*size(src,2) - 2*(L - 1)
    dst = similar(src, (dst_rows, dst_cols))
    @show size(src) size(dst)
    t = @elapsed upsample_conv_lanczos!(dst, src, base)
    alloc = @allocated upsample_conv_lanczos!(dst, src, base)
    println("lanczos elapsed: $(round(t*1000, digits=2)) ms, allocated: $alloc bytes")
end

bench()
