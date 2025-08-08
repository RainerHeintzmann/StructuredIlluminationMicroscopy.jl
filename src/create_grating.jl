# A Julia Version of calculating the grating parameters to be used on SLMs
# This file is part of the Julia version of the python code by Christian Karras

"""
    generate_grating(grating_para, phase_nr, num_phases; dim_slm=(1280, 1024), method="binary", blaze=0, periods=1, binary_threshold=0)

Generate a grating pattern for SLM given grating parameters.

Arguments:
- grating_para: Vector or Matrix. If 1D, [ahx, ahy, apx, apy]. If 2D, each row is [ahx, ahy, apx, apy].
- phase_nr: Integer, current phase index.
- num_phases: Integer, total number of phases.
- dim_slm: Tuple, SLM dimensions (default (1280, 1024)).
- method: "binary" or "full".
- blaze: Blaze amplitude (default 0).
- periods: Number of periods for phase steps (default 1).
- binary_threshold: Threshold for binary grating (default 0).

Returns:
- Grating array.
"""
function generate_grating(grating_para, phase_nr, num_phases; DT=Float32, dim_slm=(1280, 1024), method="binary", blaze=0, periods=1, binary_threshold=0, phase_factor=1)
    period = DT.(calc_per(grating_para[1,:], grating_para[2,:], grating_para[3,:], grating_para[4,:]))
    direction = DT.(calc_orient(grating_para[3,:], grating_para[4,:]))
    kx = DT.(reorient(2π .* sin.((direction .* π) ./ 180) ./ period, Val(3)))
    ky = DT.(reorient(2π .* cos.((direction .* π) ./ 180) ./ period, Val(3)))
    sz = dim_slm # (dim_slm..., size(kx,1)
    x_kx_y_ky = xx(DT,sz) .* kx .+ yy(DT,sz) .* ky
    myphase = phase_nr * phase_factor * π * periods / num_phases + 1e-4
    grating = sin.(x_kx_y_ky .+ DT(myphase))
    if blaze >= 0
        add_blaze = DT.(blaze .* mod.((x_kx_y_ky .+ DT(myphase)) ./ (2π), DT(1)))
    else
        add_blaze = DT.(abs(blaze) .* (1 .- mod.((x_kx_y_ky .+ DT(myphase)) ./ (2π), DT(1))))
    end
    # end

    if method == "binary"
        grating .= (grating .> binary_threshold) .+ add_blaze
    else
        grating .= (grating .+ 1) ./ 2 .+ add_blaze
    end
    return grating
end

# using ImageIO
# using ColorTypes

"""
    save_grating_im(path, num_phase, dim_slm, para_set; circle_rad=-1, name_ext="", test_grating_shift=false, bitdepth=1, form="png", val=(0,255), method="binary", blaze=0, func=1)

Saves grating images for all phases, optionally with a circular mask and shifted test images.
"""
function save_grating_im(path, num_phase, dim_slm, para_set; circle_rad=-1, name_ext="", test_grating_shift=false, bitdepth=1, form="png", val=(0,255), method="binary", blaze=0, func=1)
    # Create and save bright image
    myzero = ones(reverse(dim_slm))
    bright_img = UInt8.(myzero .* val[2])
    save(joinpath(path, "bright.$form"), bright_img)

    img = nothing
    for phase_nr in 0:num_phase-1
        grat = generate_grating(para_set[3:6], phase_nr, num_phase; dim_slm=dim_slm, method=method, blaze=blaze)

        # Optionally test grating shift
        if test_grating_shift
            test_grat(path, grat, para_set[3:6], "TESTSHIFTw$(para_set[1])a$(Int(para_set[2]))p$phase_nr.$form", phase_nr)
        end

        # midpos = dim_slm .÷2 .+1
        # Apply circular mask if requested
        if circle_rad != -1
            # mask = disc(dim_slm, circle_rad; offset=midpos .+ (0, 0))
            mask = create_circle_mask(dim_slm, maskpos=(0,0), radius=circle_rad)
            grat .*= mask
        end

        name = "$(name_ext)_w$(para_set[1])a$(Int(para_set[2]))p$phase_nr.$form"
        if size(grat, 3) > 1
            @error "Grating has more than one channel! Please check the input parameters."
        end
        img = UInt8.(transpose(grat[:,:,1] .* (val[2]-val[1]) .+ val[1]))
        if bitdepth == 1
            img = img .> 127 # map(x -> x > 127 ? 255 : 0, img)
        elseif bitdepth == 8
            # already UInt8
        else
            @warn "Invalid bitdepth! Either 1 or 8"
            return
        end
        save(joinpath(path, name), img)
    end
    return transpose(img)
end

"""
    load_grat_para_file(para_path; version=2)

Load a parameter file for grating generation.

# Arguments
- `para_path`: Path to the parameter file.
- `version`: Version of the file format (default: 2).

# Returns
- `Paras`: Matrix of parameters (4 x N or N x 4, depending on file).
- `NumPhases`: Number of phases (integer).
"""
function load_grat_para_file(para_path; version=2)
    # Read all lines to extract NumPhases from line 7 (index 6)
    lines = readlines(para_path)
    l = lines[7]  # Julia is 1-based, so line 7 is index 6 in Python
    NumPhases = parse(Int, strip(l[15:end]))  # Python: l[14:], Julia: l[15:end] (1-based)

    # Load parameter matrix, skipping header lines
    if version < 2
        Paras = readdlm(para_path, Float64, skipstart=16) # using DelimitedFiles
    elseif version == 2
        Paras = readdlm(para_path, Float64, skipstart=17)
    else
        error("Unsupported version: $version")
    end

    return Paras, NumPhases
end

"""
    create_grating(para_path, save_grat_folder; dim_slm=(2048,1536), circle_aperture_radius=-1,
        name_tag="", test_grating_shift=false, bitdepth=1, form="png", val="max", method="binary",
        version=2, blaze_vec=nothing, func=1)

Creates grating images using a parameter text file produced by "find_grating".

Arguments:
- para_path: Path to parameter text file.
- save_grat_folder: Folder to save images.
- dim_slm: SLM pixel size (tuple).
- circle_aperture_radius: Radius of circular aperture mask (pixels).
- name_tag: Specifier for image names.
- test_grating_shift: Save shifted grating images.
- bitdepth: 1 or 8.
- form: Image format.
- val: Value range ("max", int, or tuple).
- method: "binary" or "full".
- version: Integer to track versions.
- blaze_vec: List of blaze values per direction.
- func: 1 for 0-π phase shift, 2 for 0-2π phase shift.
"""
function create_grating(para_path, save_grat_folder; dim_slm=(2048,1536), circle_aperture_radius=-1, name_tag="", 
                        test_grating_shift=false, bitdepth=1, form="png", val="max", method="binary", version=2,
                        blaze_vec=nothing, func=1)
    # Helper for value checking
    check_val(val) = clamp(val, 0, 255)

    # Bitdepth and value handling
    if bitdepth == 1
        if method != "binary" || val != "max"
            @warn "Bitdepth is 1 -> creating binary image between 0 and 1"
        end
        val = (0,255)
        method = "binary"
    else
        if val == "max"
            val = (0,255)
        elseif isa(val, Integer)
            val = (0, check_val(val))
        elseif isa(val, Tuple) || isa(val, AbstractVector)
            val = (check_val(val[1]), check_val(val[2]))
        else
            val = (0,255)
        end
    end

    # Load parameters and number of phases
    Paras, NumPhases = load_grat_para_file(para_path, version=version)

    # Ensure output folder exists
    if !isdir(save_grat_folder)
        mkpath(save_grat_folder)
    end

    # Blaze vector handling
    if blaze_vec === nothing
        blaze_vec = fill(0, size(Paras,1))
    elseif isa(blaze_vec, AbstractVector)
        blaze_vec, Paras = adjust_lists(blaze_vec, Paras)
    else
        throw(TypeError("Wrong type for blaze_vec -> should be nothing or Vector"))
    end

    img = nothing
    # Save gratings
    for i in 1:size(Paras,1)
        p = Paras[i,:]
        blaze_val = blaze_vec[i]
        @info "Parameters: $p"
        img = save_grating_im(save_grat_folder, NumPhases, dim_slm, p; circle_rad=circle_aperture_radius, name_ext=name_tag, 
                        test_grating_shift=test_grating_shift, bitdepth=bitdepth, form=form, val=val, method=method, blaze=blaze_val, func=func)
    end
    writedlm(joinpath(save_grat_folder, "Blaze_info.txt"), blaze_vec)
    return img  # blaze_vec
end

# You will need to implement or adapt:
# - load_grat_para_file
# - adjust_lists
# - save_grating_im
# - StructuredIlluminationToolbox.generate_grating / generate_grating2
