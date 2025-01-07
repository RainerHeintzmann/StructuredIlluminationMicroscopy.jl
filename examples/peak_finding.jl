# some test to determine which peak-finding algorithm is best suited for SIMParams
using StructuredIlluminationMicroscopy.IndexFunArrays
using NDTools
using StructuredIlluminationMicroscopy.SeparableFunctions

sz = (512,)

kg = 17.767 
signal = cos.(2π * kg/sz[1] .* xx(sz))


k0 = round(kg)
exp1 = exp.(1im * 2π * k0/sz[1] .* xx(sz.-1))
exp2 = exp.(1im * 2π * k0/sz[1] .* (xx(sz.-1) .+ 1))
# exp2 = exp.(1im * 2π * k0/(sz[1]+2) .* xx(sz .+2))
# exp3 = exp.(1im * 2π * k0/sz[1] .* (xx(sz) .+ 0.5))
# winb = window_hanning(sz)
win = window_hanning(sz.-1)

@vv win
s1 = collect(ft(signal))
s2 = collect(ft(exp1))
cro = collect(select_region(ft(exp2), sz))
@vt s1' s2' cro'

# @vt ft(vcat(win.*signal,0))
# @vt ft((win.*signal)[1:end-1])

ang1 = angle(sum(exp1 .* signal[1:end-1] .* win))
ang2 = angle(sum(exp2 .* signal[2:end] .* win))
# ang1 = angle(sum(exp1 .* (winb .* signal)[1:end-1]))
# ang2 = angle(sum(exp2 .* (winb .* signal)[2:end]))
# ang2 = angle(sum(exp2[1:end-2] .* signal .* win))
# ang3 = angle(sum(exp3 .* signal .* win))

kgp = k0 - (sz[1]-1)*(ang2-ang1)/(2pi)    #*sz[1]/2π

function subpixel_kg(data, k0; dim=1)
    # @show k_off
    sz = size(data)
    alldims = ntuple(d -> (d<dim) ? d : d+1, ndims(data)-1)
    k_mod = k0 # ntuple(d->(d != dim) ? k0[d] : k0[d] + k_off, ndims(data))

    my_nd_exp = exp_ikx_sep(sz, shift_by=k_mod)
    # @vv ft(my_nd_exp .* data)
    proj = sum(my_nd_exp .* data, dims=alldims)
    # sum1 = zero(eltype(data))
    # sum2 = zero(eltype(data))
    # exp_factor = 1.0 # exp.(1im * 2π * k_off/sz[dim]) # k0[dim]
    # cur_exp = 1.0 # exp_factor
    # for a in firstindex(proj):lastindex(proj)-1
    #     # sum1 += cur_exp * proj[a] * win[a]
    #     # cur_exp *= exp_factor
    #     # sum2 += cur_exp * proj[a+1] * win[a]
    #     sum1 += proj[a] * win[a]
    #     sum2 += proj[a+1] * win[a]
    # end

    # exp1 = exp.(1im * 2π * k0/sz[1] .* xx(sz.-1))
    # exp2 = exp.(1im * 2π * k0/sz[1] .* (xx(sz.-1) .+ 1))
    # ang1 = angle(sum(exp1 .* data[1:end-1] .* win))
    # ang2 = angle(sum(exp2 .* data[2:end] .* win))
    win = window_hanning((sz[dim],).-1) # needs a tuple as size input
    sum1 = sum(proj[1:end-1] .* win)
    sum2 = sum(proj[2:end] .* win)
    # kgp = k0 - (sz[1]-1)*(ang2-ang1)/(2pi)    #*sz[1]/2π
    kgp = k0[dim] + (sz[dim]-1)*(angle(sum2)-angle(sum1))/(2pi)    #*sz[1]/2π
    return kgp
end

function subpixel_kg(data, k0)
    sz = size(data)
    my_nd_exp = exp_ikx_sep(sz, shift_by=k0)
    kg = zeros(length(k0))
    for dim=1:ndims(data)
        alldims = ntuple(d -> (d<dim) ? d : d+1, ndims(data)-1)
        proj = sum(my_nd_exp .* data, dims=alldims)
        win = window_hanning((sz[dim],).-1) # needs a tuple as size input
        sum1 = sum(proj[1:end-1] .* win)
        sum2 = sum(proj[2:end] .* win)
        kg[dim] = k0[dim] + (sz[dim]-1)*(angle(sum2)-angle(sum1))/(2pi)    #*sz[1]/2π
    end
    return kg
end

kg = 100.267 
sz = (512,)
signal = exp.(2π * 1im * kg/sz[1] .* xx(sz))
k0 = round(kg)
# kgp = subpixel_kg(signal, k0, win, dim=1, k_off=0)
kgp = subpixel_kg(signal, k0, win, dim=1)

sz = (512,512)
# kg = (17.767, 19.234)
kg = (110.267, 102.760)
signal = cos.(2π * kg[1]/sz[1] .* xx(sz)) .* cos.(2π * kg[2]/sz[2] .* yy(sz))
# @vv ft(signal)
k0 = round.(kg)
# k0 = (110, 100)
win = window_hanning(sz.-1)
kg = subpixel_kg(signal, k0)

kg = subpixel_kg(signal, k0, dim=1)
kgp = subpixel_kg(signal, k0, dim=2)
