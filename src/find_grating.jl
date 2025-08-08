# A Julia Version of calculating the grating parameters to be used on SLMs
# This file is part of the Julia version of the python code by Christian Karras

mutable struct PARA_SET
    wavelength::Float64
    angle::Float64
    para_list::Array{Float64,2}
    opt_para::Vector{Float64}
    opt_ratio::Float64
    valid::Bool
    sum_ratio::Union{Float64, Vector{Float64}} # For storing the ratio of unwanted orders
    # sum_ratio is either a single Float64 (for 1D para_list) or a Vector{Float64} (for 2D para_list)
    # This allows for flexible handling of the parameter list size.


    # ps = PARA_SET(0.488, 0.1)
    function PARA_SET(wavelength::Float64=0.0, angle::Float64=0.0, para_list=Array{Float64,2}(undef, 4, 0))
        opt_para = Float64[]
        opt_ratio = 0.0
        valid = false
        sum_ratio = 0.0
        new(wavelength, angle, para_list, opt_para, opt_ratio, valid, sum_ratio)
    end
end

function disp(ps::PARA_SET)
    println()
    println("Elements of parameter set:")
    println("==========================")
    println()
    println("Wavelength: ", ps.wavelength)
    println("Angle: ", ps.angle)
    println("Size of parameter array: $(size(ps.para_list, 2))")
end

function get_first_period(ps::PARA_SET)
    if size(ps.para_list, 2) == 0
        @warn "Parameter list is empty"
        return nothing
    elseif size(ps.para_list, 2) == 1
        pl = ps.para_list
    else
        pl = ps.para_list[:, 1]
    end
    return calc_per(pl...)
end

function filter_sum_ratios!(ps::PARA_SET, num_phase, num_dir, dim_slm, generation, pxs, f, h; square=true)
    # If para_list is a vector (1D)
    if ndims(ps.para_list) == 1
        ps.sum_ratio = get_ratio_summed_imgs(ps.para_list, num_phase, num_dir, dim_slm, generation, pxs, f, ps.wavelength, h, square)
        if ps.sum_ratio < 0.05
            # keep as is
        else
            ps.para_list = Array{Float64,1}(undef, 0)
        end
    # If para_list is a matrix (2D)
    elseif ndims(ps.para_list) == 2
        ps.sum_ratio = zeros(size(ps.para_list, 2))
        for (count, paras) in enumerate(eachcol(ps.para_list))
            ps.sum_ratio[count] = get_ratio_summed_imgs(paras, num_phase, num_dir, dim_slm, generation, pxs, f, ps.wavelength, h, square)
        end
        condition = ps.sum_ratio .< 0.05
        index_list = findall(condition)
        ps.para_list = ps.para_list[:, index_list]
    end
    return ps
end

function get_ratio_unwanted_order!(ps::PARA_SET, w_gauss, px_size, h, dim_slm, f, num_dir, num_phase; DT=Float32, dbg=-1)
    # Accept scalar or vector for w_gauss
    println("Finding ratio of wanted and unwanted orders for: angle = $(ps.angle) and wavelength = $(ps.wavelength) nm")
    println("Creating Illumination pattern and masks ...")
    if length(w_gauss) == 1
        w_gauss = (w_gauss, w_gauss)  # in case that only one number given -> so two different directions possible!
    end
    sigma = DT.(w_gauss .* 1e4 ./ (2 .* px_size))
    MyIllu = collect(gaussian(DT, dim_slm; sigma=sigma))

    # Compute masks
    MyMasks = generate_mask(num_dir, ps.angle, dim_slm, h, ps.wavelength, get_first_period(ps), px_size; f=f)
    # @show sum(MyMasks[1])
    # @show sum(MyMasks[2])
    # show(stdout, "text/plain",Integer.(MyMasks[1][400+3-10:400+3+10, 300+149-10:300+149+10]))
    # println();
    # show(stdout, "text/plain",Integer.(MyMasks[2][400+3-10:400+3+10, 300+149-10:300+149+10]))
    # println();
    println("Generating Gratings and computing FT")
    # if ndims(ps.para_list) == 1
    #     grats = generate_grating(ps.para_list, 0, num_phase; dim_slm=dim_slm) .* MyIllu
    #     MyFT = ft2d(grats)
    # else
    #     @show size(MyIllu)
    q = generate_grating(ps.para_list, 0, num_phase; DT=DT, dim_slm=dim_slm)
    # @show q[1:10,1:10,1]
    # grats = permutedims(q, (2,3,1)) .* MyIllu
    grats = q .* MyIllu
    MyFT = abs.(ft2d(grats)) # ./ sqrt(DT(prod(size(q)[1:2]))))
    # NOTE: Masks are somehow still a tiny bit different than in the python code.
    # @show sum(abs2.(q[:,:,1]))
    # @show sum(abs2.(MyIllu))
    # @show sum(abs2.(grats[:,:,1]))
    # @show sum(abs2.(MyFT[:,:,1]))
    # @show MyFT[400+3, 300+149, 1]
    # end

    # println("Computing Ratios...")
    Wanted = MyFT .* MyMasks[1]
    Unwanted = MyFT .* MyMasks[2]
    Wanted_sum = sum(Wanted, dims=(1,2))
    Unwanted_sum = sum(Unwanted, dims=(1,2))
    # end
    Ratio = Unwanted_sum ./ Wanted_sum

    # if ndims(ps.para_list) == 1
    #     ps.opt_para = ps.para_list
    #     ps.opt_ratio = Ratio
    # else
    idx = argmin(Ratio)[3] # since a CartesianIndex is returned
    ps.opt_para = ps.para_list[:, idx]
    ps.opt_ratio = minimum(Ratio)
    # end
    println("Ratio of Unwanted and Wanted orders: $(minimum(Ratio))")

    # Optional debug output
    if dbg != -1
        # Add debug output and visualization as needed
        println("Debug mode not implemented in Julia translation.")
    end

    return Ratio
end

function get_ratio_summed_imgs(paras, num_phase, num_dir, dim_slm, generation, pxs, f, wl, h; square=true)
    # Check minimum SLM size
    for d in dim_slm
        if d < 300
            error("For computing the ratio the minimum dimension for the SLM is 300x300 pixels")
        end
    end

    per = calc_per(paras...)
    ang = calc_orient(paras[3:end]...)
    im = nothing
    for i in 0:num_phase-1
        g = generation == 1 ? generate_grating(paras, i, num_phase; dim_slm=dim_slm) :
                              generate_grating(paras, i, num_phase; dim_slm=dim_slm, phase_factor=2)
        if im === nothing
            im = g
        else
            im = cat(im, g; dims=3)
        end
    end

    MyMask_W, MyMask_UW = generate_mask(num_dir, ang, dim_slm, h, wl, per, pxs, f, use_zero=false)
    mask = MyMask_W

    # Apply mask and Fourier transform
    ftims = ft2d(im)
    filtered = mask .* ftims
    fim = ift2d(filtered, ret="real")
    if square
        fim .^= 2
    end
    # Crop edges
    fim = fim[101:dim_slm[1]-100, 101:dim_slm[2]-100, :]

    ptp = maximum(fim) - minimum(fim)
    s = sum(fim, dims=1)
    s_ptp = maximum(s) - minimum(s)
    return s_ptp / ptp
end

function calc_orient(apx, apy)
    # apx = Float64.(apx)
    # apy = Float64.(apy)
    # phi = atan.(apy, apx) .* 180 ./ π
    phi = ifelse.(apx .== 0, 90, atan.(apy./apx) .* 180 ./ π)

    # phi = replace_nan.(phi, 90.0)
    return phi
end

function replace_nan(arr, val)
    arr = copy(arr)
    for i in eachindex(arr)
        if isnan(arr[i])
            arr[i] = val
        end
    end
    return arr
end

function calc_per(ahx, ahy, apx, apy)
    # All inputs should be numbers or arrays
    ahx = Float64.(ahx)
    ahy = Float64.(ahy)
    apx = Float64.(apx)
    apy = Float64.(apy)
    orient_ap = calc_orient(apx, apy)
    orient_ah = calc_orient(ahx, ahy)
    return sqrt.(ahx.^2 .+ ahy.^2) .* abs.(sin.((orient_ap .- orient_ah) .* π / 180))
end

function check_phase_steps(num_phase, para; PhaseCheckMethod=2)
    # para: 4 x N matrix (ahx, ahy, apx, apy)
    ahx = para[1, :]
    ahy = para[2, :]
    apx = para[3, :]
    apy = para[4, :]

    if PhaseCheckMethod == 2
        apx = apx .÷ gcd.(apx, apy)
        apy = apy .÷ gcd.(apx, apy)
        vert = (ahx .== 0) .* (ahy .% num_phase .== 0) .+ (ahx .!= 0) .* ((lcm.(abs.(ahx), abs.(apx)) .* (apy ./ apx .- ahy ./ ahx)) .% num_phase .== 0)
        hor = (ahy .== 0) .* (ahy .% num_phase .== 0) .+ (ahy .!= 0) .* ((lcm.(abs.(ahy), abs.(apy)) .* (apx ./ apy .- ahx ./ ahy)) .% num_phase .== 0)
    elseif PhaseCheckMethod == 1
        vert = (ahx .== 0) .* (ahy .% num_phase .== 0) .+ (ahx .!= 0) .* ((lcm.(abs.(ahx), abs.(apx)) .* apy ./ apx .- ahy ./ ahx) .% num_phase .== 0)
        hor = (ahy .== 0) .* (ahy .% num_phase .== 0) .+ (ahy .!= 0) .* ((lcm.(abs.(ahy), abs.(apy)) .* apx ./ apy .- ahx ./ ahy) .% num_phase .== 0)
    else
        error("Wrong PhaseStepMethod: Must be 1 or 2")
    end

    vert = replace_nan_bool(vert)
    hor = replace_nan_bool(hor)
    idx = findall(x -> x, (hor .+ vert) .!= 0)
    return para[:, idx]
end

function replace_nan_bool(arr)
    arr = copy(arr)
    for i in eachindex(arr)
        if isnan(arr[i])
            arr[i] = false
        end
    end
    return arr
end

"""
    This function searches a given pixelspace from start to end in steps of one for phi_x, phi_y, hx and hy
    and returns an array consisting of arrays [ahx, ahy, apx, apy] that contain the pixel values in order to fullfill the grating condition
    and the phase step condition
    
    use fixed_angle_set to define if the angle parameter shall be fixed!
            -> either give -1 (no fixed angle set) or [apx, apy]
"""
function check_for_matching_k(start::Int, stop::Int, num_phases::Int, k::Float64, dk::Float64; fixed_angle_set=nothing, PhaseCheckMethod=2)
    # Define pixel space to analyze for gratings
    ah_x = collect(start:stop-1)
    ah_y = collect(start:stop-1)
    if isnothing(fixed_angle_set)
        ap_x = collect(start:stop-1)
        ap_y = collect(start:stop-1)
    else
        ap_x = [fixed_angle_set[1]:fixed_angle_set[1]]
        ap_y = [fixed_angle_set[2]:fixed_angle_set[2]]
    end

    # Create meshgrid and flatten
    APY, APX, AHX, AHY = ndgrid(ah_x, ah_y, ap_x, ap_y)
    ahx = vec(AHX)
    ahy = vec(AHY)
    apx = vec(APX)
    apy = vec(APY)

    # Compute grating period
    # rng = 13947:13950
    period = calc_per(ahx, ahy, apx, apy)

    # Find parameter combinations that produce the correct grating period
    condition = (abs.(period) .> (k - dk/2)) .& (abs.(period) .< (k + dk/2))
    index_list = findall(condition)
    ahx = ahx[index_list]
    ahy = ahy[index_list]
    apx = apx[index_list]
    apy = apy[index_list]

    # Stack results into 4 x N matrix
    res = hcat(ahx, ahy, apx, apy)'
    # res = reshape(res, length(index_list), 4)

    # Fulfill phase step condition
    res2 = check_phase_steps(num_phases, res; PhaseCheckMethod=PhaseCheckMethod)
    return res2
end

# Helper for ndgrid (Julia equivalent of numpy.meshgrid with indexing='ij')
function ndgrid(x, y, z, w)
    X = reshape(x, :, 1, 1, 1)
    Y = reshape(y, 1, :, 1, 1)
    Z = reshape(z, 1, 1, :, 1)
    W = reshape(w, 1, 1, 1, :)
    return (repeat(X, 1, length(y), length(z), length(w)),
            repeat(Y, length(x), 1, length(z), length(w)),
            repeat(Z, length(x), length(y), 1, length(w)),
            repeat(W, length(x), length(y), length(z), 1))
end

"""
    search_direction(start_ang, d_ang, num_dir, para; fixed_angle_set=nothing)

Searches for parameter sets in specified directions based on the start angle and angular step size.

Parameters:
+ start_ang: Starting angle in degrees
+ d_ang: Angular step size in degrees
+ num_dir: Number of directions to search
+ para: 4 x N matrix (ahx, ahy, apx, apy)
+ fixed_angle_set: Optional fixed angle set to search for specific angles

Returns a vector of matrices containing the parameter sets for each direction.
"""
function search_direction(start_ang, d_ang, num_dir, para; fixed_angle_set=nothing)
    # para: 4 x N matrix (ahx, ahy, apx, apy)
    apx = para[3, :]
    apy = para[4, :]
    para_list = Vector{Matrix{Float64}}()

    # println("Searching for directions with start angle: $start_ang and angular step size: $d_ang")

    if isnothing(fixed_angle_set)
        orientation = calc_orient(apx, apy)
        for i in 0:num_dir-1
            angle = i * 180 / num_dir + start_ang
            if angle > 90
                angle -= 180
            end
            condition = (orientation .> (angle - d_ang / 2)) .& (orientation .< (angle + d_ang / 2))
            index_list = findall(condition)
            res = para[:, index_list]
            push!(para_list, res)
        end
    else
        condition = (apx .== fixed_angle_set[1]) .& (apy .== fixed_angle_set[2])
        index_list = findall(condition)
        res = para[:, index_list]
        push!(para_list, res)
    end
    return para_list
end

"""
    This function returns the coordinate map of the fourier image in the back focal plane

    i.e.: If the object is placed in the front focal plane, the image in the back focal plane is the fourier transform of the image
            The each object coorinate (unit [lenght]) maps to an angle/frequency in the back focal plane (unit [1/length])
            This function transforms the angles into length distances in the fourier plane

        M is the image (or the function)
        pxs is the pixelsize in the given direction
        wavelength is the wavelenghts of the light

        Note: the units of pxs and wavelength have to be identical!!! (e.g. um)

    focal_length  - The focal length of the imageing lens

        Note: the result will have the same unit as the focal_length (e.g. cm)


    axis is the axis in which the ramp points
    shift: use true if the ft is shifted (default setup)
    real: use true if it is a real ft (default is false)

    Example:
            if the image is a image of 20 um pixelsize of alternating black and white
            (e.g. a grating pattern of grating period 40 um), the wavelengths is 488 nm
            and the focal length is 500 mm, the peaks are located at +-6.1 mm around
            the center. This distance is givne here!


"""
function bfp_coords(sz; pxs, wavelength, focal_length, axis=1)
    return ramp(axis, sz[axis]) * wavelength * focal_length /  (sz[axis] * pxs) # , scale=ScaFT
end

"""
    This function returns the frequency ramp along a given axis of the image M

    M is the image (or the function)
    pxs is the pixelsize in the given direction
        Note: the unit of the frequency ramp is 1/unit of pxs

Parameters
im: image to generate the frequency ramp for
pxs: pixelsize
shift: use true if the ft is shifted (default setup)
real: use true if it is a real ft (default is false)
axis: is the axis in which the ramp points

returns the frequency ramp

Example:
    if you have an image with a pixelsize of 80 nm, which is 100 pixel along the axis you wanna create the ramp
    you will get a ramp runnig up to 0.006125 1/nm in steps of 0.0001251  1/nm

See also:
    applyPhaseRamp()
"""
# function freq_ramp(sz; pxs=50, shift=true, real=false, axis=1)
#     return ramp(axis, sz, scale=ScaFT) / (sz[axis] * pxs)
# end

"""
    create_circle_mask(mysize =(256,256),;askpos = (0,0), radius=100, zero = "center")
CREATE_CIRCLE_MASK creates a circle mask of given radius around the maskpos coordinates
:param mysize: tuple of sizes ()
:param maskpos: center of circle
:param radius: circle radius -> can be list or tuple if the radii in two directions are different (i.e. elliptical mask)
:param zero: 'center': cooridnate origin at center
            'image': like image coordinates (zero is in the upper left corner)
:return: a 2-D mask

Example:
```julia
> out = create_circle_mask((256,256),(0,0), 100, "center")
```
"""
function create_circle_mask(mysize =(256,256);maskpos = (0,0), radius=100, zero = "center")
    if isa(radius, Number)
        radius = (radius, radius)
    end
    if (zero == "center")
        midpos = mysize .÷2 .+1
        return disc(mysize, radius; offset=midpos .+ maskpos)
        # xr = xx(mysize)
        # yr = yy(mysize)
    elseif (zero == "image")
        return disc(mysize, radius; offset=maskpos)
        # xr = xx(mysize, offset= CtrCorner)
        # yr = yy(mysize, offset= CtrCorner)
    end
    # mask = (abs2.(xr.-maskpos[1])./abs2.(radius[1])+abs2.(yr.-maskpos[2])./abs2.(radius[2]).<1).*1
    # return mask
end

"""
Generate the fouriermask:
num_dir: number of directions
start_dir: desired start direction
dim_slm: dimensions of the slm (required for pixel size of the mst) in pixel
h_diameter: hole diameter in mm
wl: wavelength in nm
period: grating period in pixelsize
pixel_size in um

f: focal lenght of collimating lens in mm Standard is 300 mm
zero: zeros order?

The output mask is in pixels in the Fourierspace
"""
function generate_mask(num_dir, start_dir, dim_slm, h_diameter, wl, period, pixelsize; f=300, use_zero=false)
    # bfp_coords should return coordinates in back focal plane (mm)
    bfp_xx = bfp_coords(dim_slm, pxs=pixelsize, wavelength=wl/1000, focal_length=f, axis=1)
    bfp_yy = bfp_coords(dim_slm, pxs=pixelsize, wavelength=wl/1000, focal_length=f, axis=2)
    bfp_px_size = ((maximum(bfp_xx) - minimum(bfp_xx)) / size(bfp_xx, 1),
                   (maximum(bfp_yy) - minimum(bfp_yy)) / size(bfp_yy, 2))
    # bfp_px_size = ((np.max(bfp_xx)-np.min(bfp_xx))/bfp_xx.shape[0],(np.max(bfp_yy)-np.min(bfp_yy))/bfp_yy.shape[1]);

    d = wl / 1000 * f / (period * pixelsize) # distance in BFP (mm)
    disc_radius = h_diameter ./ (2 .* bfp_px_size)

    # Wanted transmission mask
    x_pos = d * sin(start_dir * π / 180) / bfp_px_size[1]
    y_pos = d * cos(start_dir * π / 180) / bfp_px_size[2]
    # Unwanted transmission mask
    mask_wanted_dir = zeros(Bool, dim_slm)
    mask_unwanted_dir = zeros(Bool, dim_slm)

    # midpos = dim_slm .÷2 .+1
    # mask_wanted_dir = collect(disc(dim_slm, disc_radius; offset=midpos .+ (x_pos, y_pos)))
    mask_wanted_dir .= collect(create_circle_mask(dim_slm, maskpos=(x_pos, y_pos), radius=disc_radius))
    # mask_wanted_dir .+= disc(dim_slm, disc_radius; offset=midpos .+ (-x_pos, -y_pos))
    mask_wanted_dir .+= create_circle_mask(dim_slm, maskpos=(-x_pos, -y_pos), radius=disc_radius)

    for i in 1:num_dir-1
        x_pos = d * sin((start_dir + i*180 / num_dir) * π / 180) / bfp_px_size[1]
        y_pos = d * cos((start_dir + i*180 / num_dir) * π / 180) / bfp_px_size[2]
        # mask_unwanted_dir .+= disc(dim_slm, disc_radius; offset = midpos .+ (x_pos, y_pos))
        mask_unwanted_dir .+= create_circle_mask(dim_slm, maskpos=(x_pos, y_pos), radius=disc_radius)
        # mask_unwanted_dir .+= disc(dim_slm, disc_radius; offset = midpos .+ (-x_pos, -y_pos))
        mask_unwanted_dir .+= create_circle_mask(dim_slm, maskpos=(-x_pos, -y_pos), radius=disc_radius)
    end
    if use_zero
        # mask_unwanted_dir .+= disc(dim_slm, disc_radius; offset=midpos .+ (0, 0))
        mask_unwanted_dir .+= create_circle_mask(dim_slm, maskpos=(0, 0), radius=disc_radius)
    end
    return (mask_wanted_dir, mask_unwanted_dir)
end

function clear_para_list(pl, al, n_dir, lam)
    # pl: Vector of PARA_SET
    # al: Vector of angles
    # n_dir: number of directions
    # lam: Vector of wavelengths

    if !isempty(al)
        wl_set = Set(lam)
        new_l = PARA_SET[]
        for ang in al
            ang_set = Set([ang + n * 180 / n_dir for n in 0:n_dir-1])
            ang_set_ok = true
            for angle in ang_set
                if any(x -> x.angle == angle, pl)
                    ang_set_ok &= true
                else
                    ang_set_ok &= false
                end
            end
            append!(new_l, [p for p in pl if ang_set_ok && (p.angle in ang_set)])
        end

        anglist = Float64[]
        for p in new_l
            if !(p.angle in anglist)
                push!(anglist, p.angle)
            end
        end

        new_l2 = PARA_SET[]
        for el in anglist
            sub_list = filter(x -> x.angle == el, new_l)
            wl_set_ok = true
            for wavelength in wl_set
                if any(x -> x.wavelength == wavelength, sub_list)
                    wl_set_ok &= true
                else
                    wl_set_ok &= false
                end
            end
            if wl_set_ok
                append!(new_l2, sub_list)
            end
        end
    else
        new_l2 = PARA_SET[]
    end
    return new_l2
end

"""
    create_para_list(start, stop, k0, dk0_R, d_ang, lam, num_dir, num_phase;
                        fixed_angle=[-1], fixed_angle_set=nothing, PhaseCheckMethod=2)

Parameters:
- start: Start pixel for the search range of the unit cell
- stop: Stop pixel for the search range of the unit cell
- k0: Desired grating period in pixels
- dk0_R: Relative tolerance for the grating period
- d_ang: Angular step size for the search in degrees
- lam: Array of Wavelengths to consider
- num_dir: Number of directions to search
- num_phase: Number of phases to consider

Example:
```julia
using StructuredIlluminationMicroscopy
> para_list, angle_array, error = create_para_list(1, 100, 3.1, 0.3, 60, [0.488], 1, 7)
```
"""
function create_para_list(start, stop, k0, dk0_R, d_ang, lam, num_dir, num_phase; # , dim_slm, h, generation, px_size, f, opt_grating_sum
                        fixed_angle=[-1], fixed_angle_set=nothing, PhaseCheckMethod=2)
    # Determine angle array
    if fixed_angle == [-1]
        angle_array = collect(1:1:(180/num_dir)-1) # 180 scan whole start angle range in 1 degree steps
    else
        angle_array = fixed_angle
    end

    if !isnothing(fixed_angle_set) 
        @info "Grating vector describing the angle was fixed to $fixed_angle_set"
        angle = calc_orient(fixed_angle_set[1], fixed_angle_set[2])
        angle_array = [angle]
    end

    para_list = PARA_SET[]
    error = 0
    for wl in lam
        k = k0 * wl / lam[1]
        @info "Searching parameter sets for wavelength: $wl nm"
        @info "Desired grating period: $k pixels"
        @info "Searching grating period ..."

        res = check_for_matching_k(
            start, stop, num_phase, k, k * dk0_R;
            fixed_angle_set=fixed_angle_set, PhaseCheckMethod=PhaseCheckMethod
        )
        @info "$(size(res, 2)) sets found with matching grating period"
        @info "Scanning start angles and searching for directions..."

        angle_arr = Float64[]
        for start_ang in angle_array
            # start_ang = 0.0
            liste = search_direction(start_ang, d_ang, num_dir, res; fixed_angle_set=fixed_angle_set)
            tester = all(.!isempty.(liste))
            if tester
                push!(angle_arr, start_ang)
                for i in 0:num_dir-1
                    new_ang = start_ang + i * 180 / num_dir
                    push!(para_list, PARA_SET(Float64(wl), new_ang, liste[i+1]))
                end
            end
        end

        angle_array = angle_arr
        if isempty(angle_arr)
            @warn "NO MATCHING GRATINGS FOUND: RAISE SEARCH RANGE"
            error = -1
            break
        else
            @info "$(length(angle_arr)) possible starting angles found:"
            @info angle_arr
        end
    end

    return clear_para_list(para_list, angle_array, num_dir, lam), angle_array, error
end

"""
    optimize_grating_sum(pl, num_phase, num_dir, dim_slm, generation, pxs, f, h; square=true)

Optimizes PARA_SETs by filtering out parameter sets with high unwanted order ratios (sum_ratio ≥ 0.05).
Returns a filtered vector of PARA_SETs.

Arguments:
- pl: Vector of PARA_SET objects
- num_phase: Number of phases
- num_dir: Number of directions
- dim_slm: Dimensions of the SLM (default (1024,1024))
- generation: Generation type (1 or 2)
- pxs: Pixel size (μm)
- f: Focal length (mm)
- h: Hole diameter (mm)
- square: If true, uses squared images for ratio calculation (default true)
Returns:
- Vector of PARA_SETs with valid para_list after filtering

Example:
```julia
using StructuredIlluminationMicroscopy
> ps = PARA_SET(0.488, 0.1)
> pl = [ps]
> optimized_pl = optimize_grating_sum(pl, 7, 3, (1024, 1024), 1, 0.1, 300, 0.1)
```
"""
function optimize_grating_sum(pl, num_phase, num_dir, dim_slm, generation, pxs, f, h; square=true)
    new_pl = PARA_SET[]
    for p in pl
        filter_sum_ratios!(p, num_phase, num_dir, dim_slm, generation, pxs, f, h; square=square)
        # Only keep PARA_SETs with valid para_list after filtering
        if ndims(p.para_list) == 1
            if length(p.para_list) > 0
                push!(new_pl, p)
            end
        elseif ndims(p.para_list) == 2
            if size(p.para_list, 2) > 0
                push!(new_pl, p)
            end
        end
    end
    return new_pl
end

"""
    find_optimum_set(pl, w_gauss, px_size, h, angle_array, num_dir, num_phases; dim_slm=(1024,1024), f=300, criterion="maximum")

Finds the optimum set of PARA_SETs for grating generation, minimizing unwanted orders.

Arguments:
- pl: Vector of PARA_SET objects
- w_gauss: Gaussian beam radius (cm or vector)
- px_size: pixel size (μm)
- h: hole diameter (mm)
- angle_array: array of starting angles
- num_dir: number of directions
- num_phases: number of phases

- dim_slm: SLM dimensions (default (1024,1024))
- f: focal length (mm)
- criterion: "maximum" or "average"

Returns:
- Vector of PARA_SETs with optimum angle

Example:
```julia
using StructuredIlluminationMicroscopy
> numdirs = 1
> pl, angle_array, error = pl, angle_array, error = create_para_list(1, 100, 3.1, 0.3, 11, [0.488], numdirs, 7)
> optimum_set = find_optimum_set(pl, 2.0, 12.0, 0.1, [0, 45, 90], 3, 7)
```

"""
function find_optimum_set(pl, w_gauss, px_size, h, angle_array, num_dir, num_phases; dim_slm=(1024,1024), f=300, criterion="maximum")
    average_list = Float64[]
    max_list = Float64[]
    # Compute unwanted order ratios for each PARA_SET
    for p in pl
        get_ratio_unwanted_order!(p, w_gauss, px_size, h, dim_slm, f, num_dir, num_phases)
    end
    # For each angle, collect stats
    for angle in angle_array
        new_list = filter(x -> mod(x.angle, 180 ÷ num_dir) ≈ angle, pl)
        if !isempty(new_list)
            push!(average_list, mean(getfield.(new_list, :opt_ratio)))
            push!(max_list, maximum(getfield.(new_list, :opt_ratio)))
        end
    end
    # Find optimum angle
    if criterion == "average"
        opt_angle = angle_array[argmin(average_list)]
    else # "maximum"
        opt_angle = angle_array[argmin(max_list)]
    end
    println()
    println("Optimum starting angle: $opt_angle")
    println("Maximum ratio unwanted orders: $(minimum(max_list))")
    println("Average ratio unwanted orders: $(minimum(average_list))")
    println("====================================================")
    opt_list = filter(x -> mod(x.angle, 180 ÷ num_dir) ≈ mod(opt_angle, 180 ÷ num_dir), pl)
    for el in opt_list
        println("$(el.wavelength) ; $(el.angle) ; $(el.opt_para)")
    end
    return opt_list
end

"""
    create_grating_param_file(path; start=-10, end=50, num_dir=3, num_phase=3, wavelength=[488, 561, 638, 405], error_period=0.01, error_angle=0.1, fixed_angle=nothing, px_size=8.2, w_gauss=0.5, h=0.3, dim_slm=(1024,1024), f=250, NA=1.46, magnification=76.8, BFP_filling=83.4, period=nothing, fixed_angle_set=nothing, PhaseCheckMethod=2, optimize_for_unwanted_orders=true, opt_grating_sum=false, generation=1)

Generate a parameter set for SLM gratings and save to a text file.
Returns the path to the parameter file.
"""
function create_grating_param_file(path; 
    start=-10, stop=50, num_dir=3, num_phase=3, wavelength=[488, 561, 638, 405], error_period=0.01, error_angle=0.1,
    fixed_angle=nothing, px_size=8.2, w_gauss=0.5, h=0.3, dim_slm=(1024,1024), f=250, NA=1.46, magnification=76.8,
    BFP_filling=83.4, period=nothing, fixed_angle_set=nothing, PhaseCheckMethod=2, optimize_for_unwanted_orders=true,
    opt_grating_sum=false, generation=1
)
    lam = collect(wavelength)
    fixed_angle_val = isnothing(fixed_angle) ? [-1] : [fixed_angle]

    if !isnothing(fixed_angle_set) 
        @info "Fixed angle set given -> only 1 Direction possible! -> resetting num_dir to 1"
        num_dir = 1
    end

    if (isnothing(period) && isnothing(BFP_filling)) || (!isnothing(period) && !isnothing(BFP_filling))
        println("Error in defining period: Either period or BFP_filling has to be given! Not both!")
        return "ERROR"
    else
        if isnothing(period)
            period = get_period(lam[1], BFP_filling/100, px_size, NA, magnification)
        else
            try
                BFP_filling = lam[1]*magnification/(1000*px_size*period*NA)*100
            catch
                BFP_filling = nothing
            end
        end

        error = 0
        error_period_px = period * error_period / 100
        println("Error Period in pixels: $error_period_px")

        name = "para_$(round(period, digits=3))_$(num_phase)phases_$(num_dir)Dir.txt"

        para_list, angle_array, error = create_para_list(
            start, stop, period, error_period_px, error_angle, lam, num_dir, num_phase;
            # dim_slm, h, generation, px_size, f, opt_grating_sum; 
            fixed_angle=fixed_angle_val, fixed_angle_set=fixed_angle_set, PhaseCheckMethod=PhaseCheckMethod
        )

        if opt_grating_sum
            para_list = optimize_grating_sum(para_list, num_phase, num_dir, dim_slm, h, generation, px_size, f)
        end

        para_list = clear_para_list(para_list, angle_array, num_dir, lam)
        if isempty(para_list)
            error = -1
            @error "No elements after para list creation"
        end

        if error == 0
            if optimize_for_unwanted_orders
                ol = find_optimum_set(para_list, w_gauss, px_size, h, angle_array, num_dir, num_phase, dim_slm=dim_slm, f=f)
            else
                ol = para_list
            end
            save_gratings(ol, period, error_period_px, num_dir, num_phase, px_size, w_gauss, h, dim_slm, error_angle, f, path, BFP_filling, name, optimized_list=optimize_for_unwanted_orders)
        else
            @error "could not find parameter set"
        end
        return ol
        # return joinpath(path, name)
    end
    return ol
end

function save_gratings(ol, period, error_period, num_dir, num_phase, px_size, w_gauss, h, dim_slm, error_angle, f, path, bfp_fill, name="para_data.txt"; optimized_list=true)
    # Compose metadata header
    ts = now()
    st = Dates.format(ts, "yyyy-mm-dd HH:MM:SS")
    meta_text = "Gratings computed with Julia translation of Christian's Python code\n"
    meta_text *= st * "\n\n"
    meta_text *= "Desired Period ($(ol[1].wavelength)nm) :\t$(period) Pixels\n"
    meta_text *= "Maximum Error Period:\t$(error_period) Pixels\n"
    meta_text *= "Maximum Error Angle:\t$(error_angle) Degree\n"
    meta_text *= "Number Phases:\t$(num_phase)\n"
    meta_text *= "Number Directions:\t$(num_dir)\n"
    meta_text *= "Pixel size:\t$(px_size) um\n"
    meta_text *= "Spot diameter on SLM (1/e^2 radius, gaussian):\t$(w_gauss) cm\n"
    meta_text *= "Diameter of holes in mask:\t$(h) mm\n"
    meta_text *= "Focal length of collimating lens:\t$(f) mm\n"
    meta_text *= "Dimensions SLM:\t$(dim_slm[1]) X $(dim_slm[2]) Pixels\n"
    meta_text *= "BFP Filling for first wavelength:\t$(bfp_fill)% \n\n"
    meta_text *= "Parameters (Check Ronny's Publication to understand [OPT EXPR Vol22 No17 2014]) \n"
    meta_text *= "Wavelength [nm]\tangle [DEG]\th_x [px]\th_y [px]\ttheta_x [px]\ttheta_y [px]\tRatio of unwanted orders\tperiod\tangle\n"

    for p in ol
        if optimized_list
            meta_text *= string(p.wavelength) * "\t" * string(p.angle) * "\t" *
                         string(p.opt_para[1]) * "\t" * string(p.opt_para[2]) * "\t" *
                         string(p.opt_para[3]) * "\t" * string(p.opt_para[4]) * "\t" *
                         string(p.opt_ratio) * "\t" *
                         string(calc_per(p.opt_para[1], p.opt_para[2], p.opt_para[3], p.opt_para[4])) * "\t" *
                         string(calc_orient(p.opt_para[3], p.opt_para[4])) * "\n"
        else
            if ndims(p.para_list) == 1
                meta_text *= string(p.wavelength) * "\t" * string(p.angle) * "\t" *
                             string(p.para_list[1]) * "\t" * string(p.para_list[2]) * "\t" *
                             string(p.para_list[3]) * "\t" * string(p.para_list[4]) * "\t-1\t" *
                             string(calc_per(p.para_list...)) * "\t" *
                             string(calc_orient(p.para_list[3], p.para_list[4])) * "\n"
            elseif ndims(p.para_list) == 2
                for i in 1:size(p.para_list, 2)
                    p2 = p.para_list[:, i]
                    meta_text *= string(p.wavelength) * "\t" * string(p.angle) * "\t" *
                                 string(p2[1]) * "\t" * string(p2[2]) * "\t" *
                                 string(p2[3]) * "\t" * string(p2[4]) * "\t-1\t" *
                                 string(calc_per(p2...)) * "\t" *
                                 string(calc_orient(p2[3], p2[4])) * "\n"
                end
            end
        end
    end

    # Write metadata to file
    open(joinpath(path, name), "w") do io
        write(io, meta_text)
    end

    # Optionally, save a "bright" image and all gratings as TIFFs (not implemented here)
    # You can add code here to generate and save the actual grating images if needed.

    return nothing
end

function get_period(lam, eta, pixelpitch=8.2, NA=1.46, magnification=76.8)
    # lam in nm, eta (0-1), pixelpitch in um, NA, magnification
    period = lam * magnification / (1000 * pixelpitch * eta * NA)
    return period
end