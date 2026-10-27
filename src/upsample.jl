# upsampling

# """
#     upsample_conv2!(dst, src, kx_even, kx_odd, ky_even, ky_odd)
# Upsample 2D `src` into `dst` (size 2Nx×2Ny) via separable odd/even convolutions.

# Parameters:
# - `dst`: destination array, size `(2*size(src,1), 2*size(src,2))`
# - `src`: source array to upsample
# - `kx_even`: 1D kernel along x for even x-pixels
# - `kx_odd`:  1D kernel along x for odd x-pixels
# - `ky_even`: 1D kernel along y for even y-pixels
# - `ky_odd`:  1D kernel along y for odd y-pixels
# """
# function upsample_conv2!(dst, src, kx_even, kx_odd, ky_even=kx_even, ky_odd=kx_odd)
#     dst .= 0
#     # Center kernels with OffsetArrays so index variable runs from -(c-1):+(L-c)
#     cx_even = (length(kx_even) + 1) ÷ 2
#     cx_odd  = (length(kx_odd)  + 1) ÷ 2
#     cy_even = (length(ky_even) + 1) ÷ 2
#     cy_odd  = (length(ky_odd)  + 1) ÷ 2

#     Kx_even = OffsetArray(kx_even, -(cx_even-1):(length(kx_even)-cx_even))
#     Kx_odd  = OffsetArray(kx_odd,  -(cx_odd-1):(length(kx_odd)-cx_odd))
#     Ky_even = OffsetArray(ky_even, -(cy_even-1):(length(ky_even)-cy_even))
#     Ky_odd  = OffsetArray(ky_odd,  -(cy_odd-1):(length(ky_odd)-cy_odd))

#     # # Convolve along x producing two fields (even/odd), same size as src.
#     # convx_even = @tullio tmpx[n, m] := src[n + j, m] * kx_even[j] pad=true
#     # convx_odd  = @tullio tmpx[n, m] := src[n + j, m] * kx_odd[j]  pad=true

#     # Interleave along x into a mid array of size (2Nx, Ny)
#     # mid = similar(dst, size(src,1)*2, size(src,2))
#     vx_odd = @view dst[1:2:end,1:2:end];
#     vx_even =  @view dst[2:2:end,1:2:end];
#     vy_odd = @view dst[:,1:2:end];
#     vy_even =  @view dst[:,2:2:end];
#     # @tullio vx_odd[n, m] = src[n + j - 1, m] * kx_odd[j] 
#     # @tullio vx_even[n, m] = src[n + j - 2, m] * kx_even[j]
#     @tullio vx_odd[n, m] = src[n + j, m] * Kx_odd[j] 
#     @tullio vx_even[n, m] = src[n + j, m] * Kx_even[j]

#     # @tullio mid_odd[n, m] := src[n + j, m] * Kx_odd[j] pad=true
#     # @tullio mid_even[n, m] := src[n + j, m] * Kx_even[j] pad=true
#     # mid[1:2:end, :] .= mid_odd
#     # mid[2:2:end-1, :] .= mid_even

#     # return vx_odd
#     # Convolve along y on the mid array, producing two fields (even/odd)
#     @tullio vy_even[n, m] = vy_odd[n, m + q] * Ky_even[q] pad=true
#     @tullio vy_odd[n, m] = vy_odd[n, m + q] * Ky_odd[q] pad=true
#     # @tullio midy_odd[n, m] := mid[n, m + q] * Ky_odd[q] pad=true
#     # @tullio midy_even[n, m] := mid[n, m + q] * Ky_even[q] pad=true

#     # Interleave along y into the destination array (2Nx × 2Ny)
#     # dst[:, 2:2:end-1] .= midy_even
#     # dst[:, 1:2:end] .= midy_odd
#     @show size(vy_even)
#     dst[:, 2:2:end] .= vy_even
#     @show size(vy_odd)
#     dst[:, 1:2:end] .= vy_odd

#     return dst
# end

#### another implementation with KernelAbstractions
using KernelAbstractions, StaticArrays

using Tullio

function conv_upsample_tullio(x, w_poly)
# Pre-split your 5x5 kernel into 4 sub-phases (e.g., 3x3 each)
# w_poly[ki, kj, phase_i, phase_j]
# phase_i/j are 1 or 2 (Julia is 1-indexed)

    @tullio y[pi, i + _, pj, j + _] := x[i+ki, j+kj] * w_poly[ki, kj, pi, pj]

    return reshape(y, (size(y,1)*size(y,2),size(y,3)*size(y,4)))
end

# Der Kernel für eine Dimension (kann für X oder Y genutzt werden)
@kernel function separable_conv_kernel!(output, input, kernel_1d, ::Val{dim}) where dim
    # Globaler Index
    idx = @index(Global, NTuple)
    I = CartesianIndex(idx)
    
    # Kernel-Radius berechnen
    K = length(kernel_1d)
    half_k = K ÷ 2
    
    # Akkumulator für die Faltung
    acc = zero(eltype(input))
    
    # Faltung entlang der gewählten Dimension (dim=1 -> horiz, dim=2 -> vert)
    for k in 1:K
        # Offset berechnen
        offset = k - (half_k + 1)
        
        # Erstelle den Nachbar-Index
        neighbor_idx = I + offset * unit_vector(Val(dim))
        
        # Boundary Check (einfaches Zero-Padding)
        if checkbounds(Bool, input, neighbor_idx)
            acc += input[neighbor_idx] * kernel_1d[k]
        end
    end
    
    output[I] = acc
end

# Der Kernel für eine Dimension (kann für X oder Y genutzt werden)
"""
@kernel function separable_conv_upsample!(output, input, kernel_1d, ::Val{stride}, ::Val{offset}, ::Val{dim}) where dim

this kernel convolves and upsamples the data.
"""
@kernel function separable_conv_upsample!(output, input, kernel_1d_even, kernel_1d_odd, ::Val{dim}) where dim
    # Globaler Index
    idx = @index(Global, NTuple)
    I = CartesianIndex(idx)
    I2e = CartesianIndex(stretch_vector(Val(dim)) .*idx)
    I2o = CartesianIndex(stretch_vector(Val(dim)) .*idx .+ unit_vector_t(Val(dim)))
    
    # Kernel-Radius berechnen
    K = length(kernel_1d_even)
    half_k = K ÷ 2
    
    # Akkumulator für die Faltung
    acc_even = zero(eltype(input))
    acc_odd = zero(eltype(input))
    
    # Faltung entlang der gewählten Dimension (dim=1 -> horiz, dim=2 -> vert)
    for k in 1:K
        # Offset berechnen
        offset = k - (half_k + 1)
        
        # Erstelle den Nachbar-Index
        neighbor_idx = I + offset * unit_vector(Val(dim))
        
        # Boundary Check (einfaches Zero-Padding)
        if checkbounds(Bool, input, neighbor_idx)
            acc_even += input[neighbor_idx] * kernel_1d_even[k]
            acc_odd += input[neighbor_idx] * kernel_1d_odd[k]
        end
    end
    
    output[I2e] = acc_even
    if checkbounds(Bool, output, I2o)
        output[I2o] = acc_odd
    end
end

# Hilfsfunktion für Richtungsvektoren
unit_vector(::Val{1}) = CartesianIndex(1, 0)
unit_vector(::Val{2}) = CartesianIndex(0, 1)
unit_vector_t(::Val{1}) = (1, 0)
unit_vector_t(::Val{2}) = (0, 1)
stretch_vector(::Val{1}) = (2,1) # CartesianIndex(2, 1)
stretch_vector(::Val{2}) = (1,2) # CartesianIndex(1, 2)

function run_convolution!(dst, src, kernel_vec)
    # Wähle Backend basierend auf dem Datentyp (Array -> CPU, CuArray -> GPU)
    backend = KernelAbstractions.get_backend(src)
    output_tmp = similar(dst)

    # Kernel vorbereiten
    kernel_fn = separable_conv_kernel!(backend, 16) # 16 ist die Workgroup-Größe
    
    # 1. Horizontaler Pass (dim=1)
    kernel_fn(output_tmp, src, kernel_vec, Val(1), ndrange=size(src))
    KernelAbstractions.synchronize(backend)
    
    # 2. Vertikaler Pass (dim=2)
    kernel_fn(dst, output_tmp, kernel_vec, Val(2), ndrange=size(output_tmp))
    KernelAbstractions.synchronize(backend)

    return dst
end

function run_conv_upsample!(dst, src, kernel_vec_even, kernel_vec_odd; wg=16)
    # Wähle Backend basierend auf dem Datentyp (Array -> CPU, CuArray -> GPU)
    backend = KernelAbstractions.get_backend(src)
    us_size_tmp = ntuple(d->(d==1) ? size(dst,d) : size(src,d), ndims(src))
    output_tmp = similar(dst, us_size_tmp)

    # Kernel vorbereiten
    kernel_fn = separable_conv_upsample!(backend, wg) # 16 ist die Workgroup-Größe
    
    @assert all(size(dst)[1:2] .>= size(src)[1:2].*2)
    @assert length(kernel_vec_even) == length(kernel_vec_odd)

    # 1. Horizontaler Pass (dim=1)
    kernel_fn(output_tmp, src, kernel_vec_even, kernel_vec_odd, Val(1), ndrange=size(src))
    KernelAbstractions.synchronize(backend)

    # return output_tmp
    # 2. Vertikaler Pass (dim=2)
    kernel_fn(dst, output_tmp, kernel_vec_even, kernel_vec_odd, Val(2), ndrange=size(output_tmp))
    KernelAbstractions.synchronize(backend)

    return dst
end


function run_convolution(data, kernel_vec)    
    output_final = similar(data)    
    return run_convolution!(output_final, data, kernel_vec)
end

function run_conv_upsample(data, kernel_vec_even, kernel_vec_odd; wg=16)
    newsize = ntuple(d -> size(data,d)*(1 + (d<3)), ndims(data))
    output_final = similar(data, newsize)
    return run_conv_upsample!(output_final, data, kernel_vec_even, kernel_vec_odd; wg=wg)
end

# upsample with convolution using strided views and adding
function upsample_bc!(dst, src)
    us_size_tmp = ntuple(d->(d==1) ? size(dst,d) : size(src,d), ndims(src))
    output_tmp = similar(dst, us_size_tmp)
    
    vx1 = @view src[1:end-2,:]    
        vx2 = @view src[2:end-1,:]
    vx3 = @view src[3:end,:]
    c1o = 0f0; c2o= 1f0; c3o = 0f0;
    c1e = 0f0; c2e= 0.5f0; c3e = 0.5f0;
    ov_o = @view output_tmp[1:2:end,:]
    ov_e = @view output_tmp[2:2:end,:]

    # @show size(vx1)
    # @show size(ov_o)
    ov_o .= vx1 .* c1o .+ vx2 .* c2o .+ vx3 .* c3o;
    ov_e .= vx1 .* c1e .+ vx2 .* c2e .+ vx3 .* c3e;

    # return output_tmp
    vx1 = @view output_tmp[:,1:end-2]    
    vx2 = @view output_tmp[:,2:end-1]
    vx3 = @view output_tmp[:,3:end]
    ov_o = @view dst[:,1:2:end]
    ov_e = @view dst[:,2:2:end]

    ov_o .= vx1 .* c1o .+ vx2 .* c2o .+ vx3 .* c3o;
    ov_e .= vx1 .* c1e .+ vx2 .* c2e .+ vx3 .* c3e;

    return dst
end
    """
        upsample_bc!(dst, src, k_odd, k_even)

    Efficient separable upsampling by factor 2 using explicit odd/even phase kernels.
    - `k_odd`: 1D FIR taps applied to the odd-phase outputs
    - `k_even`: 1D FIR taps applied to the even-phase outputs

    Kernels are interpreted centered at index `(length(k)+1)>>1` and used as
    linear combinations of shifted views. Broadcasting is fused via
    `Broadcast.broadcasted` + `Broadcast.materialize!` to avoid temporaries.
    """
    function upsample_bc!(dst, src, k_odd::AbstractVector, k_even::AbstractVector)
        @assert ndims(src) == 2 "upsample_bc! currently supports 2D arrays"
        @assert length(k_odd) >= 1 && length(k_even) >= 1 "Kernel lengths must be ≥ 1"

        Lodd = length(k_odd); Leven = length(k_even)
        codd = (Lodd + 1) >>> 1
        ceven = (Leven + 1) >>> 1
        offs_odd = collect((1:Lodd) .- codd)   # offsets relative to odd center
        offs_even = collect((1:Leven) .- ceven) # offsets relative to even center

        # First pass: along dim 1 (rows) -> output_tmp
        us_size_tmp = ntuple(d -> (d == 1) ? size(dst, d) : size(src, d), ndims(src))
        output_tmp = similar(dst, us_size_tmp)

        pad = max((Lodd - 1) >>> 1, (Leven - 1) >>> 1)
        rows_valid = (1 + pad):(size(src, 1) - pad)

        # Build views aligned to each offset
        views_odd = Vector{AbstractArray{eltype(src),2}}(undef, Lodd)
        views_even = Vector{AbstractArray{eltype(src),2}}(undef, Leven)
        for (i, s) in pairs(offs_odd)
            views_odd[i] = @view src[(first(rows_valid) + s):(last(rows_valid) + s), :]
        end
        for (i, s) in pairs(offs_even)
            views_even[i] = @view src[(first(rows_valid) + s):(last(rows_valid) + s), :]
        end

        ov_o = @view output_tmp[1:2:end, :]
        ov_e = @view output_tmp[2:2:end, :]

        _materialize_lincomb!(ov_o, views_odd, k_odd)
        _materialize_lincomb!(ov_e, views_even, k_even)

        # Second pass: along dim 2 (cols) -> dst
        cols_valid = (1 + pad):(size(output_tmp, 2) - pad)
        views_odd = Vector{AbstractArray{eltype(output_tmp),2}}(undef, Lodd)
        views_even = Vector{AbstractArray{eltype(output_tmp),2}}(undef, Leven)
        for (i, s) in pairs(offs_odd)
            views_odd[i] = @view output_tmp[:, (first(cols_valid) + s):(last(cols_valid) + s)]
        end
        for (i, s) in pairs(offs_even)
            views_even[i] = @view output_tmp[:, (first(cols_valid) + s):(last(cols_valid) + s)]
        end

        ov_o = @view dst[:, 1:2:end]
        ov_e = @view dst[:, 2:2:end]

        _materialize_lincomb!(ov_o, views_odd, k_odd)
        _materialize_lincomb!(ov_e, views_even, k_even)

        return dst
    end

    # SVector convenience overloads
    function upsample_bc!(dst, src, k_odd::SVector, k_even::SVector)
        return upsample_bc!(dst, src, collect(k_odd), collect(k_even))
    end

    # Helper: fused broadcasted linear combination of views and coefficients
    function _materialize_lincomb!(out, views, coeffs)
        @assert length(views) == length(coeffs)
        @views begin
            bc = Broadcast.broadcasted(*, views[1], coeffs[1])
            for i in 2:length(coeffs)
                bc = Broadcast.broadcasted(+, bc, Broadcast.broadcasted(*, views[i], coeffs[i]))
            end
            Broadcast.materialize!(out, bc)
        end
        return out
    end

function upsampled_size(src, kodd, keven=kodd)
    Lmax = max(length(kodd), length(keven))
    dst_rows = 2*size(src,1) - 2*(Lmax - 1)
    dst_cols = 2*size(src,2) - 2*(Lmax - 1)
    return (dst_rows, dst_cols)
end

"""
    upsample_conv_lanczos!(dst, src, kernel)

Upsample `src` into `dst` by factor 2 using separable Lanczos interpolation
with distinct odd/even phase taps derived from a single base `kernel`.

Steps:
- Build Lanczos taps of length `length(kernel)` for phase 0 (odd) and 0.5 (even),
  with support `a = (length(kernel)-1)/2`.
- Normalize each Lanczos vector to sum to 1.
- Elementwise multiply with `kernel` to form `k_odd`, `k_even`.
- Call `upsample_bc!(dst, src, k_odd, k_even)`.
"""
function upsample_conv_lanczos!(dst, src, kernel::AbstractVector)
    @assert ndims(src) == 2 "upsample_conv_lanczos! supports 2D arrays"
    k_odd, k_even = get_lanczos_conv_kernels(kernel)
    return upsample_bc!(dst, src, k_odd, k_even)
end

@inline function _sincπ(x)
    x == 0 ? one(x) : sinpi(x) / (pi*x)
end

"""
    get_lanczos_conv_kernels(kernel; a = (length(kernel)-1)/2, Lout = length(kernel))

Construct odd/even real-space kernels that jointly perform interpolation (Lanczos)
and convolution by multiplying their frequency responses, then returning
time-domain taps cropped to approximately the original length.

Returns `(k_odd, k_even)` with length `Lout` (default: `length(kernel)`).
"""
function get_lanczos_conv_kernels(kernel::AbstractVector; a = (length(kernel)-1)/2, Lout::Int = length(kernel))
    Lk = length(kernel)
    T = eltype(kernel)
    # Build Lanczos phase taps of desired output length
    c_out = (Lout + 1) >>> 1
    offs = collect(1:Lout) .- c_out
    lodd = similar(kernel, Lout)
    leven = similar(kernel, Lout)
    @inbounds for i in 1:Lout
        x0 = T(offs[i])
        xh = T(offs[i] - 0.5)
        lodd[i] = _lanczos_sample(x0, T(a))
        leven[i] = _lanczos_sample(xh, T(a))
    end
    # Normalize phase taps to unit sum (unit DC)
    sodd = sum(lodd); seven = sum(leven)
    if sodd != 0; lodd .= lodd ./ sodd; end
    if seven != 0; leven .= leven ./ seven; end

    # Frequency-domain combination: multiply spectra, then inverse to time domain
    Llin = Lk + Lout - 1
    Nfft = Llin
    pad_k = vcat(kernel, zeros(T, Nfft - Lk))
    pad_o = vcat(lodd,   zeros(T, Nfft - Lout))
    pad_e = vcat(leven,  zeros(T, Nfft - Lout))
    Hk = rfft(pad_k)
    Ho = rfft(pad_o)
    He = rfft(pad_e)
    Hodd = Hk .* Ho
    Heven = Hk .* He
    comb_odd = irfft(Hodd, Nfft)
    comb_even = irfft(Heven, Nfft)

    # Crop around the convolution center to target length Lout
    c_k = (Lk + 1) >>> 1
    c_i = (Lout + 1) >>> 1
    c_conv = c_k + c_i - 1
    start = c_conv - (Lout - 1) >>> 1
    stop  = start + Lout - 1
    @assert 1 <= start <= Nfft && stop <= Nfft "Cropping exceeds bounds"
    k_odd  = comb_odd[start:stop]
    k_even = comb_even[start:stop]

    # Preserve DC gain approximately equal to the original kernel sum
    sk = sum(kernel)
    so = sum(k_odd); se = sum(k_even)
    if so != 0; k_odd  .= k_odd  .* (sk/so); end
    if se != 0; k_even .= k_even .* (sk/se); end
    return k_odd, k_even
end
@inline function _lanczos_sample(x, a)
    if a == 0
        return x == 0 ? one(x) : zero(x)
    end
    ax = abs(x)
    if ax < a
        return _sincπ(x) * _sincπ(x/a)
    elseif ax == a
        return zero(x)
    else
        return zero(x)
    end
end
