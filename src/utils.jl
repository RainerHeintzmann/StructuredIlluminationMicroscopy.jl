"""
    weight_matrix(sp::SIMParams)

construct a weight matrix from the peak_phases and peak_strengths in the SIMParams object.

Parameters:
+ `sp::SIMParams` : SIMParams object, containing SIMParams.peak_phases and SIMParams.peak_strengths

"""
function weight_matrix(sp::SIMParams)
    return cis.(sp.peak_phases) .* sp.peak_strengths
end

"""
    pinv_weight_matrix(sp; Eps=1e-6)

return the pseudo-inverse of the weight matrix from the peak_phases and peak_strengths in the SIMParams object.

Parameters:
+ `sp::SIMParams` : SIMParams object, containing SIMParams.peak_phases and SIMParams.peak_strengths
+ `Eps::Float64` : threshold for zeroing out small values

"""
function pinv_weight_matrix(sp; Eps=1e-6)
    res = pinv(weight_matrix(sp))
    res[abs.(res) .< Eps] .= 0.0
    return res
end

"""
    rfft_size(sz)

return the size of the rfft of a given input size `sz`.

Parameters:
+ `sz` : (iterable) size of the input array

"""
function rfft_size(sz)
    return ntuple((d) -> (d==1) ? sz[d]÷ 2 + 1 : sz[d], length(sz))
end

"""
    get_rft_center(sz)
    
return the center of the shifted rfft of a given size `sz`.

"""
function get_rft_center(sz)
    ntuple((d) -> (d==1) ? 1 : sz[d] .÷ 2 .+ 1, length(sz))
end

function rfftshift(myrft)
    fftshift(myrft, 2:ndims(myrft))
end

function rfftshift!(dst, src)
    fftshift!(dst, src, 2:ndims(src))
end

function rifftshift(myrft)
    ifftshift(myrft, 2:ndims(myrft))
end

function rifftshift!(dst, src)
    ifftshift!(dst, src, 2:ndims(src))
end
"""
    rfft_crop(myrft, dsz)

crop the rfft of a real-valued Fourier-transform to the size `dsz`. The input `myrft` is assumed to be centered at [1,1,..]

Parameters:
+ `myrft` : rfft of a real-valued Fourier-transform (centered at the top-left corner of the array)
+ `dsz` : size of the cut output array after the rfft

"""
function rfft_crop(myrft, dsz)
    sz = size(myrft)
    ctr = sz .÷ 2 .+ 1
    drsz = rfft_size(dsz)
    cstart = ctr .- (drsz .÷ 2)
    cend = cstart .+ drsz .- 1
    cut_ranges = ntuple((d) -> (d==1) ? (1:drsz[d]) : (cstart[d] : cend[d]), length(dsz))
    res = rifftshift(rfftshift(myrft)[cut_ranges...])
    return res
end

"""
    shift_subpixel!(img, ordershift, peakphase)

shift the Fourier-transform of the image by subpixel values and returns the pixelshift.
"""
function shift_subpixel!(img, ordershift, prep, order_num)
    # pos = idx(eltype(img), size(img))
    # img .*= cis.(dot.(Ref(2pi .* subpixelshift ./ size(img)), pos))
    subpixel_shifters, pixelshift = (haskey(prep, :subpixel_shifters)) ? (prep.subpixel_shifters[order_num], prep.pixelshifts[order_num]) : get_shift_subpixel(img, ordershift)
    if (subpixel_shifters != 1)
        img .*= subpixel_shifters
    end
    return (pixelshift[1:2]..., 0)
end

"""
    shift_subpixel(img, ordershift)

modifies a separable function representation of a pixel-shifter (to be multiplied with the FFT) by appying fftshift to each of its separable components.

Parameters:
+ `img::AbstractArray` : image
+ `ordershift::NTuple{3, Float64}` : subpixel shift

"""
function fftshift_sep!(myseparable)
    for d in 1:length(myseparable.args)
        myseparable.args[d].parent .= fftshift(myseparable.args[d].parent)
    end
    return myseparable
end

"""
    ifftshift_sep!(myseparable)

modifies a separable function representation of a pixel-shifter (to be multiplied with the FFT) by appying ifftshift to each of its separable components.

Parameters:
+ `myseparable::SeparableFunction` : separable function

"""    
function ifftshift_sep!(myseparable)
    for d in 1:length(myseparable.args)
        myseparable.args[d].parent .= ifftshift(myseparable.args[d].parent)
    end
    # myseparable.args = ntuple((i) -> fftshift(myseparable.args[i]), length(myseparable.args))
    return myseparable
end

"""
    get_shift_subpixel(img, ordershift)

returns a separable function representation of a pixel-shifter to be multiplied with the FFT
"""
function get_shift_subpixel(img, ordershift)
    pixelshift = round.(Int, ordershift)
    subpixelshift = ordershift[1:2] .- pixelshift[1:2]
    if (norm(subpixelshift) == 0.0)
        return 1, pixelshift
    end

    mysepshift = exp_ikx_sep(typeof(img), size(img)[1:2]; shift_by= .-subpixelshift) # should be negative to agree with integer pixel shifts
    # ifftshift_sep!(mysepshift) # due to the shifing being in the iFFT space and not the FFT space
    return mysepshift, pixelshift
end

"""
    dot_mul_last_dim!(orders, sim_data, myinv, n, Eps = 1e-7)

perform the dot product of the last dimension of the SIM data with the inverse matrix of the weights.
This is one part of the matrix multiplicaton for unmixing the orders.
"""
function dot_mul_last_dim!(order, sim_data, myinv, n, Eps = 1e-7)
    RT = eltype(sim_data)
    CT = Complex{RT}
    contributing = findall(x->abs(x) .> Eps, myinv[n,:])
    sub_matrix = CT.(myinv[n, contributing])
    # mydstidx = ntuple(d->(d==ndims(sim_data)) ? (n:n) : Colon(), ndims(sim_data))
    dv = order # @view orders[mydstidx...] # destination view to write into
    w = sub_matrix[1]
    mymd = ntuple(d->(d==ndims(sim_data)) ? contributing[1] : Colon(), ndims(sim_data))
    sv = @view sim_data[mymd...]
    dv .= w.* (@view sim_data[mymd...])
    for md in 2:length(contributing)
        w = sub_matrix[md]
        mymd = ntuple(d->(d==ndims(sim_data)) ? contributing[md] : Colon(), ndims(sim_data))
        sv = @view sim_data[mymd...]
        dv .+= w.*sv
    end
end


"""
    get_result_size(sz, upsample_factor)

return the size of the result image given the input size and the upsample factor.

Parameters:
+ `sz` : (iterable) size of the input array
+ `upsample_factor` : upsample factor

"""
function get_result_size(sz, upsample_factor)
    upsample_factor = ntuple((d) -> (d<=2) ? upsample_factor : 1, length(sz))
    return ceil.(Int, sz .* upsample_factor)
end

"""
    conj_add!(dst, src)

add the complex conjugate of `src` to `dst` in place.

Parameters:
+ `dst` : destination array
+ `src` : source array

"""
function conj_add!(dst, src)
    dst .+= conj.(src)
end

"""
    add!(dst, src)

add `src` to `dst` in place.

Parameters:
+ `dst` : destination array
+ `src` : source array

"""
function add!(dst, src)
    dst .+= src
end

"""
    modify_otf(otf, sigma=0.1, contrast=1.0)

modify the OTF by multiplying it with a (One - contrast*Gaussian) function.

Parameters:
+ `otf` : OTF
+ `sigma` : standard deviation of the Gaussian
+ `contrast` : contrast of the Gaussian

"""
function modify_otf(otf, sigma=0.1, contrast=1.0)
    RT = real(eltype(otf))
    return otf .* (one(RT) .- RT(contrast) .* exp.(-rr2(RT, size(otf)[1:2], scale=ScaFT)/(2*sigma^2)))
end

"""
    swap_vals!(v1, v2)

swap the values of the views `v1` and `v2` in place without requiring memory.

Parameters:
+ `v1` : view 1
+ `v2` : view 2

"""
function swap_vals!(v1, v2)
    v1 .⊻= v2  # swap memory-free via three .xor operations
    v2 .⊻= v1
    v1 .⊻= v2 
end

IntType(::Type{Float16}) = Int16
IntType(::Type{Float32}) = Int32
IntType(::Type{ComplexF32}) = Int64
IntType(::Type{Float64}) = Int64
IntType(::Type{ComplexF64}) = Int128
IntType(::Type{Int16}) = Int16
IntType(::Type{Int32}) = Int32
IntType(::Type{Int64}) = Int64
IntType(::Type{Int128}) = Int128

"""
    fftshift!(dst, dims)

shift the FFT of `dst` in place. The dimensions to be shifted are specified by `dims`.
The sizes of the dimensions specified by `dims` must be even. The in-place shifted array is also returned.

Parameters:
+ `dst` : destination array
+ `dims` : dimensions to be shifted
"""
function fftshift_even!(dst, dims=1:ndims(dst))
    if any(isodd.(size(dst)[dims]))
        error("all dst sizes $(size(dst)) specified by dims $(dims) must be even, for this algorithm to work correctly")
    end
    dstr = reinterpret(IntType(eltype(dst)), dst)
    off = zeros(Int, ndims(dst))
    for perm in 1:2^(length(dims)-1)
        hilo = ntuple((d) -> (perm & (1 << (d-1))) != 0, length(dims)-1)
        for d in 1:length(dims)-1
            off[dims[d]] = hilo[d]*(size(dst)[dims[d]]÷2)
        end
        first_half = ntuple((d) -> (d in dims) ? (1+off[d]:size(dst)[d]÷2+off[d]) : Colon(), ndims(dst))
        second_half = ntuple((d) -> (d in dims) ? (size(dst)[d]÷2+1-off[d]:size(dst,d)-off[d]) : Colon(), ndims(dst))
        swap_vals!(view(dstr, first_half...), view(dstr, second_half...))
    end
    return dst
end