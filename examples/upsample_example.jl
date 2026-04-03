using CUDA, cuDNN
using StaticArrays
using StructuredIlluminationMicroscopy
using View5D
using NDTools

# using Flux # for the CUDA Version of NNlib
# using NNlib: DenseConvDims
using NNlib
using BenchmarkTools
using FourierTools

"""
    nnlib_upsample_conv(x, w)

Upsample 2D input `x` by factor 2 using transposed convolution (single Flux call).

`w` has shape `(kH, kW, 2, 2)` where dims 3 & 4 represent polyphase indices.
The polyphase sub-kernels are interleaved into a `(2kH, 2kW, 1, 1)` transposed-conv
kernel so that stride-2 conv_transpose applies each sub-kernel to the correct
output phase (odd-odd, even-odd, odd-even, even-even).

Output size: `((H-1)*2 + 2*kH, (W-1)*2 + 2*kW)` for input `(H, W)`.
"""
function nnlib_upsample_conv(x, w4)
    @assert ndims(x) in (2, 4) "Input must be 2D image or 4D (H,W,C,N)"

    x4 = ndims(x) == 2 ? reshape(x, size(x,1), size(x,2), 1, 1) : x
    kH, kW = size(w4, 1)÷2, size(w4, 2)÷2

    # Use ∇conv_data which is the transposed convolution (gradient of conv w.r.t. input)
    # For output = 2*input size, use padding = kH-1, kW-1
    # Formula: out = (in-1)*stride + kernel - 2*pad → 2*H = (H-1)*2 + 2*kH - 2*pad → pad = kH-1
    H, W, C, N = size(x4)
    pad_h, pad_w = kH - 1, kW - 1
    out_size = (2H, 2W, 1, N)
    cdims = NNlib.DenseConvDims(out_size, size(w4); stride=2, padding=(pad_h, pad_w), flipkernel=false)
    y4 = NNlib.∇conv_data(x4, w4, cdims)
    return dropdims(y4, dims=(3, 4))
end


"""
    nnlib_upsample_conv_sep(x, kernel_x, kernel_y)

Upsample 2D input `x` by factor 2 using separable transposed convolutions.

`kernel_x` has shape `(kH, 2)` where dim 2 contains odd/even phase taps for rows.
`kernel_y` has shape `(kW, 2)` where dim 2 contains odd/even phase taps for columns.

Performs two 1D transposed convolutions:
1. Upsample rows by 2 using kernel_x
2. Upsample columns by 2 using kernel_y

Output size: `(2H, 2W)` for input `(H, W)`.
"""
function nnlib_upsample_conv_sep(x, kernel_x, kernel_y)
    @assert ndims(x) in (2, 4) "Input must be 2D image or 4D (H,W,C,N)"
    @assert ndims(kernel_x) == 2 && size(kernel_x, 2) == 2 "kernel_x must be (kH, 2)"
    @assert ndims(kernel_y) == 2 && size(kernel_y, 2) == 2 "kernel_y must be (kW, 2)"

    x4 = ndims(x) == 2 ? reshape(x, size(x,1), size(x,2), 1, 1) : x
    H, W, C, N = size(x4)

    # --- Pass 1: Upsample rows (dim 1) by 2 using kernel_x ---
    kH = size(kernel_x, 1)
    # Interleave odd/even phase taps into a (2*kH, 1) kernel
    wx = similar(kernel_x, 2*kH)
    for i in 1:kH
        wx[2i-1:2i-1] = kernel_x[i:i, 1:1]  # odd phase
        wx[2i:2i]   = kernel_x[i:i, 2:2]  # even phase
    end
    wx4 = reshape(wx, 2*kH, 1, 1, 1)  # (kH, kW=1, C_out=1, C_in=1)
    
    pad_h = kH - 1
    out_size_1 = (2H, W, 1, N)
    cdims_x = NNlib.DenseConvDims(out_size_1, size(wx4); stride=(2,1), padding=(pad_h, 0), flipkernel=false)
    y_mid = NNlib.∇conv_data(x4, wx4, cdims_x)

    # --- Pass 2: Upsample columns (dim 2) by 2 using kernel_y ---
    kW = size(kernel_y, 1)
    # Interleave odd/even phase taps into a (1, 2*kW) kernel
    wy = similar(kernel_y, 2*kW)
    for j in 1:kW
        wy[2j-1:2j-1] = kernel_y[j:j, 1:1]  # odd phase
        wy[2j:2j]   = kernel_y[j:j, 2:2]  # even phase
    end
    wy4 = reshape(wy, 1, 2*kW, 1, 1)  # (kH=1, kW, C_out=1, C_in=1)
    
    pad_w = kW - 1
    out_size_2 = (2H, 2W, 1, N)
    cdims_y = NNlib.DenseConvDims(out_size_2, size(wy4); stride=(1,2), padding=(0, pad_w), flipkernel=false)
    y4 = NNlib.∇conv_data(y_mid, wy4, cdims_y)

    return dropdims(y4, dims=(3, 4))
end

function stride_kernel(w)
    kH, kW = size(w, 1), size(w, 2)
    w_interleaved = similar(w, 2kH, 2kW)
    for i in 1:kH, j in 1:kW
        w_interleaved[2i-1:2i-1, 2j-1:2j-1] = w[i:i, j:j, 1:1, 1:1]  # odd-odd
        w_interleaved[2i:2i, 2j-1:2j-1] = w[i:i, j:j, 2:2, 1:1]  # even-odd
        w_interleaved[2i-1:2i-1, 2j:2j]   = w[i:i, j:j, 1:1, 2:2]  # odd-even
        w_interleaved[2i:2i, 2j:2j]   = w[i:i, j:j, 2:2, 2:2]  # even-even
    end
    return reshape(w_interleaved, 2kH, 2kW, 1, 1)
end

function main_test()
    src = Float32.(reshape(1:16,4,4))
    run_convolution(src .+ 0.0, SVector(0.5,0.5))

    dst = zeros(Float32,size(src).*2 .-4)
    upsample_bc!(dst, src)

    src = zeros(10,10)
    src[5,5]=1; src[1,1]=1;
    dst = similar(src)
    @time run_convolution!(dst, src, SVector(0.25f0,1.0f0,0.25f0))

    # k_even = SVector(0f0, 0.5f0, 0.5f0) 
    k_even = SVector(0.5f0, 0.5f0, 0f0) 
    k_odd = SVector(0f0, 1.0f0, 0f0) 
    dst = similar(src, size(src) .* 2);
    run_conv_upsample!(dst, src, k_even, k_odd)
    # src = rand(Float32, 10000,10000)
    # dst = zeros(Float32,size(src).*2 .-4)
    # @time upsample_bc!(dst, src);

    kernel_4d = rand(Float32, 3,3,2,2)

    @time dst = conv_upsample_tullio(src, kernel_4d); # 700 ms

    kex = k_even
    kox = k_odd
    key = reorient(k_even, Val(2))
    koy = reorient(k_odd, Val(2))
    # kernel_4d = cat(cat(kox*koy, kox*key, dims=3), cat(kex*koy, kex*key, dims=3), dims=4)
    # kernel_4d ./= sum(kernel_4d, dims=(1,2))
    kernel_4d = cat(cat(kox*koy, kex*koy, dims=3), cat(kox*key, kex*key, dims=3), dims=4)
    # kernel_4d = SArray{Tuple{3,3,2,2}, Float32}(kernel_4d)
    src = zeros(Float32, 14,14)
    src[3,3]=1
    jk = stride_kernel(kernel_4d)
    dst = nnlib_upsample_conv(src, jk)

    kernel_2dx = cat(k_odd, k_even, dims=2)
    kernel_2dy = cat(k_odd, k_even, dims=2)
    dst = nnlib_upsample_conv_sep(src, kernel_2dx, kernel_2dy)

    # dst = similar(src)
    # @time run_convolution!(dst, src, SVector(0.25f0,1.0f0,0.25f0)); # 0.7 sec

    # src = rand(Float32, 10000,10000)
    # dst = similar(src, size(src).*2 .-4)
    # @time upsample_bc!(dst, src);   # 0.45 sec
    # @time q = rfft(dst); # 1.0 sec

    # src = rand(Float32, 10000,10000);
    # k_even = SVector(0.1f0, 0.5f0, 0.5f0, 0.1f0) 
    # k_odd = SVector(0f0, 0f0, 1.0f0, 0f0, 0f0) 

    src = rand(Float32, 3000, 3000);
    k_even = SVector(rand(Float32,5)...) 
    k_odd = SVector(rand(Float32,5)...) 
    # k_even = SVector(0f0, 0.5f0, 0.5f0) 
    # k_odd = SVector(0f0, 1.0f0, 0f0) 
    dst = similar(src, upsampled_size(src, k_odd, k_even));
    @time upsample_bc!(dst, src, k_odd, k_even); # 600 ms
    dst = similar(src, size(src) .* 2);
    @time run_conv_upsample!(dst, src, k_even, k_odd);

    dst2 = similar(src, size(src).*2 .-4);
    @time upsample_bc!(dst2, src); # 24 ms
    @assert dst ≈ dst2

    kernel = SVector(0f0, 0.5f0, 0.5f0, 0f0, 0f0)
    dst3 = similar(src, upsampled_size(src, kernel));
    @time upsample_conv_lanczos!(dst3, src, kernel); # 55 ms
    kernel2 = SVector(0f0, 0.25f0, 0.5f0, 0.25f0, 0f0)
    dst4 = similar(dst3);
    @time upsample_conv_lanczos!(dst4, src, kernel2); # 55 ms
    @vt dst3 dst4 
    # using FourierTools, NDTools
    # @vt select_region(upsample2(src), size(dst3)) 
    @time dst5 = upsample2(src); # 212 ms
    @assert dst3 ≈ dst4

    kernel_4d = SArray{Tuple{5,5,2,2}}(rand(Float32, 5,5,2,2))
    @time dst = conv_upsample_tullio(src, kernel_4d); # 0.7 sec
    kernel_4d = rand(Float32, 5,5,2,2)
    # kernel_4d = rand(Float32, 3,3,2,2)
    @time dst = nnlib_upsample_conv(src, kernel_4d); # 2 sec CPU

    src_c = cu(src);
    src_c = CUDA.rand(2000,2000);
    kernel_4d_c = cu(kernel_4d)
    # Does not Work:
    # CUDA.@time dst_c = conv_upsample_tullio(src_c, kernel_4d_c); # DOES NOT WORK in CUDA!
    jk = stride_kernel(kernel_4d_c)
    tc_sep = @belapsed CUDA.@sync dst_c = nnlib_upsample_conv($src_c, $jk) # 6 ms


    k_even = SVector(0.1f0, 0.5f0, 0.5f0, 0.1f0, 0.1f0) 
    k_odd = SVector(0.1f0, 0.1f0, 1.0f0, 0.1f0, 0.1f0) 
    kernel_2dx = cat(k_odd, k_even, dims=2)
    kernel_2dy = cat(k_odd, k_even, dims=2)

    kernel_2dxc = cu(kernel_2dx)
    kernel_2dyc = cu(kernel_2dy)
    tc_sep = @belapsed CUDA.@sync dst_c = nnlib_upsample_conv_sep($src_c, $kernel_2dxc, $kernel_2dyc) # 4.8 ms
    # dst = nnlib_upsample_conv_sep(src, kernel_2dx, kernel_2dy)


    tc_fft = @belapsed CUDA.@sync dst_c .= rfft($src_c) # 0.8 ms
    CUDA.@time dst_c = rfft(src_c); # 0.8 ms

    src_c2 = copy(src_c)
    tc_conv = @belapsed CUDA.@sync  dst_c2 = FourierTools.conv(dst_c, dst_c); # 12 ms
    tc_conv = @belapsed CUDA.@sync  dst_c = FourierTools.upsample2(src_c); # 3 ms

    dst_c = similar(src_c, size(src_c) .* 2);
    k_even_c = cu(k_even)
    k_odd_c = cu(k_odd)
    CUDA.@time run_conv_upsample!(dst_c, src_c, k_even_c, k_odd_c); # 22 mmsec,  KernelAbstraction based


    # src = cu(rand(Float32, 10000, 10000));
    k_even_c = cu(SVector(0f0, 0.5f0, 0.5f0)) 
    k_odd_c = cu(SVector(0f0, 1.0f0, 0f0))
    dst_c = similar(src_c, upsampled_size(src_c, k_odd_c, k_even_c));
    tc_bc = @belapsed CUDA.@sync  upsample_bc!($dst_c, $src_c, $k_odd_c, $k_even_c) # 2 ms

    src_c = cu(Float32.(reshape(1:16,4,4)))
    dst_c = similar(src_c, upsampled_size(src_c, k_odd_c, k_even_c));
    upsample_bc!(dst_c, src_c, k_odd_c, k_even_c)

    dst2 = similar(src, size(src).*2 .-4);
    CUDA.@time upsample_bc!(dst2, src); # 
    @assert dst ≈ dst2

    kernel = cu(SVector(0f0, 0.5f0, 0.5f0, 0f0, 0f0))
    dst3 = similar(src_c, upsampled_size(src_c, kernel));
    CUDA.@time upsample_conv_lanczos!(dst3, src, kernel); # 220 ms
    kernel2 = cu(SVector(0f0, 0.25f0, 0.5f0, 0.25f0, 0f0))
    dst4 = similar(dst3);
    tc_bc = @belapsed CUDA.@sync upsample_conv_lanczos!($dst4, $src_c, $kernel2) # 2ms
    @vt dst3 dst4 
    # using FourierTools, NDTools
    # @vt select_region(upsample2(src), size(dst3)) 
    CUDA.@time dst5 = upsample2(src); # 


end
