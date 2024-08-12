### A Pluto.jl notebook ###
# v0.19.45

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ 0257712c-cb02-4232-a5af-ddfcfb8702ed
using Pkg

# ╔═╡ 4d371c70-f023-46dc-bafa-732f50f20acf
 Pkg.activate(".")

# ╔═╡ b0216ace-c46c-42ee-b35f-8ca894e68a1e
using TestImages, PointSpreadFunctions, TestImages, StructuredIlluminationMicroscopy, ImageShow, PlutoUI, FourierTools, NDTools #packages

# ╔═╡ 4ba98557-a551-49d5-ad0f-23a8869d5462
md"""
## activating Enviorment from .examples
"""

# ╔═╡ ea10443f-6184-4d82-a1e4-519c1a748195
function show_many(args...; kwargs...)
	doabs(x) = abs.(x)
	nargs = args./maximum.(doabs.(args))
	simshow(cat(nargs...;dims=2); kwargs...)
end

# ╔═╡ 6d2ef0c6-6f36-4ab8-8bcd-e5b88daa3eef
md"""
## Set Parameter for PSF and for SIM Illumination pattern
In the following the Parameter for the PSF will be set. The Values of Lambda, the numerical aperature and the refractive index can be varied.
Afterwards the SIM Illumination pattern will be designed. 
The number of directions the illumination pattern will be projected onto the sample can be set. In this case the num_direction is 3(num.directions=3). The number of images which will be done is a product of the different directions of Illumination pattern and the Phase shifting of the Illumination pattern (num.images = num.directions * num.phases).In this case the phase shift of the pattern is set to 3. The product of direction and phases results in nine images.

Num.orders is related to the harmonic order which will be inculded in the reconstruction process. For num.order=2 the 0th and the 1th order will be considered. 

The 0th order will be blocked. The convolution of the 1th order with itself in the frequency domain will  produce 3 peaks (Oth, +/-1th). Since Intensity is a real the +1th and -1 orders are the same. Effectivly only the 0th and the 1th order needs to be counted. In the 2 and 3 image the 0th order are the same so they are not counted seperatedly. This results in total into 4 peaks in frequency space. 
 
In the function generate_peaks the matrices will be filled. 
The k.peak.pos gives the position of the four peaks in 3 dimensional space. 
The peak.strength and  and the peak.phases give a 9x4 Matix which corresponds to the nine images which four peaks. 
The otf.indices and otf.phases will be filled with 4 positions (harmonic orders) which are used to reconstruct the image. 

With the Number of Photons Slider the resolution of the image can be changed.
"""

# ╔═╡ cdbe23cb-6622-41fe-98bb-e4d581f64453
function make_3d_pattern(k_peak_pos, offset_phase=0.0)
    num_peaks = length(k_peak_pos)
    has_kz(p) = (p[3] != 0.0) 

    otf_indices = ones(Int, num_peaks)
    otf_indices[has_kz.(k_peak_pos)] .= 2
    otf_phases = zeros(Float64, num_peaks)
    otf_phases[has_kz.(k_peak_pos)] .= offset_phase
    return otf_indices, otf_phases
end

# ╔═╡ 7306bf9d-4670-4367-8a85-cf0fc1b22ff2
function generate_peaks(num_phases::Int=9, num_directions::Int=3, num_orders::Int=2, k1 = 0.9, single_zero_order=true, k1z = 0.0)
    num_peaks = num_directions * num_orders;
    if (single_zero_order)
        num_peaks -= num_directions - 1 
    end
    k_peak_pos = Array{NTuple{3, Float64}}(undef, num_peaks)
    current_peak = 1
    for d in 0:num_directions-1 # starts at zero
        for o in 0:num_orders-1 # starts at zero
            phi = 2pi*d/num_directions 
			@show phi
            k = k1 .* o
			@show k
            if (o > 0 || d==0 || single_zero_order == false)
                if (k == 0.0) # this is a zero order peak
                    k_peak_pos[current_peak] = (0.0, 0.0, 0.0);  # (d-1)*num_orders + o
                else
                    if (0 < o < num_orders -1 ) # this is a medium order peak, but not the highest one
                        k_peak_pos[current_peak] =  (k .*cos(phi), k .*sin(phi), k1z)
						#@show k_peak_pos
                    else
                        k_peak_pos[current_peak] = k .* (cos(phi), sin(phi), 0.0)
						@show k_peak_pos
                    end
                end
                current_peak += 1
           end
        end
    end
    peak_phases = zeros(num_phases, num_peaks)
    peak_strengths = zeros(num_phases, num_peaks)
    phases_per_cycle = num_phases ÷ num_directions
    current_peak = 1
    for p in 0:num_phases-1 # starts at zero
        current_d = p ÷ phases_per_cycle  # directions start at 0
        current_peak = 1
        for d in 0:num_directions-1 # starts at zero
            for o in 0:num_orders-1
                if (o > 0 || d==0 || single_zero_order == false)
                    if (o == 0 || d == current_d)
                        peak_phases[p+1, current_peak] = mod(2pi*o*p/phases_per_cycle, 2pi)
                        peak_strengths[p+1, current_peak] = 1.0
                    end
                    current_peak += 1
                end
            end
        end
    end
    
    otf_indices, otf_phases = make_3d_pattern(k_peak_pos, 0.0); # no offset_phase

    return k_peak_pos, peak_phases, peak_strengths, otf_indices, otf_phases
end

# ╔═╡ 4c2372be-50bc-11ef-33f1-cbf44b3e823b
begin
    lambda = 0.532; NA = 1.0; n = 1.52
    pp = PSFParams(lambda, NA, n);  # 532 nm, NA 0.25 in Water n= 1.33
    sampling = (0.06, 0.06, 0.1)  # 100 nm x 100 nm x 200 nm

    # SIM illumination pattern
    num_directions = 3; num_images =  3*num_directions; num_orders = 2
    rel_peak = 0.40 # peak position relative to sampling limit on fine grid
    k_peak_pos, peak_phases, peak_strengths, otf_indices, otf_phases = 	generate_peaks(num_images, num_directions, num_orders, rel_peak / (num_orders-1))
end

# ╔═╡ 792280f3-d803-45c8-aa96-ff329f399e9e
@show peak_strengths

# ╔═╡ 6b7e4fd9-4ad0-4105-ba18-6b7c63c77528
@bind num_photons Slider(1:1000)

# ╔═╡ 99e542ee-b981-4f2a-88cb-4da222d33e63
md"""
## Load Object and SIM Parameter
SIMParams saves the parameter in a fix structure. This way it is better organised and more efficient. The parameter are saved at spf.
The object which should be illuminated and reconstructed by SIM is loaded by testimage and saved as Float32 in obj. The function size(obj) provides the dimension of the object as a tuble (example 512x512). The . gives elementvise operations. The ÷ devides the dimension by 2 (example 256x256). The .+1 adds one to the tuble (example 257x257). The pixel value will be set equal to 2. 

"""

# ╔═╡ ce6b9058-e27a-45ed-a216-fb02d06778f2
begin
    spf = SIMParams(pp, sampling, Float64(num_photons), 100.0, k_peak_pos, peak_phases, peak_strengths, otf_indices, otf_phases);

    obj = Float32.(testimage("resolution_test_512"));
	#obj = select_region(obj, (256,256); center=(200,300))
    obj[(size(obj).÷2 .+1)...] = 2.0;
end;


# ╔═╡ e7656c27-a54d-4679-aa86-e32462456931
md"""
## Simulate Structured Ilumination
Define the downsample.factor as 2. The downsampling parameter reduces the resolution by a factor of 2. 
The function simulate.sim will be called. The function will use the parameter of obj, pp, spf and the just defined downsample.factor. The function does a simulation of the structured illuminaton microscopy. The result of the matrix will be saved in sim_data. The SIM data is a 128x128x9 Matrix. 

"""

# ╔═╡ 3be84092-c031-4988-90f0-5bd1e6ac425a
begin
   	downsample_factor = 2
	sim_data, sp = simulate_sim(obj, pp, spf, downsample_factor);
end;


# ╔═╡ 0150b9ec-8e40-4e10-bd79-e67795f7d25b
md"""
## "detected" Structured Illumination data
Since Sim_data has 3 Dimensional Data in x,y and z where z is the image number, it is possible to go with the slider through the z dimension (Simulated Images). All the components of x and y are taken into account(:).
"""

# ╔═╡ 89034637-b722-4cb9-928b-696fd5283797
@bind z_plane Slider(1:9) 

# ╔═╡ ff70b285-f07c-487f-ad36-0989985babe0
simshow(sim_data[:,:,z_plane])

# ╔═╡ d832a4c0-2356-4eb9-b758-2a6bc6cbe1b7
md"""
## Parameter for reconstruction
In the following parameters for the reconstrution are defined. For example the parameter suppression_strength gives information about the supression unwanted frequency components. Further more the the Wiener filter, which reduces noise, is here as well defined. The value of the Wiener filter can be set in the slider. 
"""

# ╔═╡ b890f350-ed74-407c-b390-bc4398a2e9fd
begin
	rp = ReconParams() ;# just use defaults
    rp.upsample_factor = 2; # 1 means no upsampling
    # rp.wiener_eps = 1e-4
    rp.suppression_strength = 0.99;
    rp.suppression_sigma = 5e-2;
    rp.do_preallocate = true;
    rp.use_measure=true;
    rp.double_use=true; rp.preshift_otfs=true; 
    rp.use_hgoal = true;
    rp.hgoal_exp = 0.5;
end

# ╔═╡ 28e982ee-90d4-42a1-9e5f-d9e4ce171d07
md"""
## Reconstruction
First the there are Preparation data for the reconstruction done. Therefore Sim.data,pp,sp and rp are used.
Recon.Sim does the reconstruction of the structured Illumination microscopy. The nine Images which where taken before are used to do a high resolution image (Matrix of 256x256). For the recon_sim the simulated structured illumination data of nine images(sim.data) is used as well as the sp data and prep of the SIM. The result is saved in variable "recon".
"""

# ╔═╡ d4875f35-52f2-4a47-b709-35bb9202cb82
md"""
## Reconstruction of Widefield Microscopy
In the follwoing step a Widefiled Microscopy image should be for comparisson generated. The sum(dim.data, dims=3) sums up the third dimension of the sim.data. In that case all the pictures which where taken before are sumed up into one picture. The size of the matrix is 128x128x1. Size(recon) gives the information of the size  which it should be formed (256x256) into and the function resample forms the new size of the array. 
"""

# ╔═╡ a1c2a41b-265b-44d2-9622-520d48f2a9eb
md"""
## Object, Widefield & SIM reconstructin
In the following Object-, Widefield-, SIM-reconstruction and OTF images are presented.
"""

# ╔═╡ 7ab1ec5b-84e0-4a2c-9062-4221844e613b
#show_many(obj, abs.(wf), abs.(recon), abs.(ft(recon)).^0.4)

# ╔═╡ a2dd8bb6-62ed-4e01-890f-113c475568d9
md"""In the next picture the original object can be seen."""

# ╔═╡ ca5390fc-6f4f-4b98-b7cd-f61291bbc5f3
simshow(obj)

# ╔═╡ 07b42d3d-374f-403b-b3c0-9af63cf31ea8
md"""
In the left picture a  Widefield microscopy image of the object is shown and on the right side its corresponding Fourier Transfer Function. 
"""

# ╔═╡ c00b653d-ede5-45ef-b3b3-4e4527d9f35f
md"""
On the left side is the reconnstructed SIM image shown. On the right side the Fourier Transformation of the reconstructed image can be seen. Its notable that the OTF is a convolution of several Widefield OTFs.
"""

# ╔═╡ 3c1c1dd2-ca10-48e1-bfaa-a476b21d5f1a
@bind wiener_slider Slider(1:10) 

# ╔═╡ 459d774a-ba1a-4dc2-bbb7-b0d8e460ac99
rp.wiener_eps = 10.0^(-wiener_slider);

# ╔═╡ ae8ea2d1-68ad-427a-b346-5ca1332f8e37
begin
	# print("Wiener Filter Parameter: 10^$(wiener_slider)")
	qq = wiener_slider
	prep = recon_sim_prepare(sim_data, pp, sp, rp); # do preallocate
end

# ╔═╡ 8051f5db-452f-4d45-8102-57892552e45d
recon = recon_sim(sim_data, prep, sp);

# ╔═╡ f7ac031c-8d9d-4cfa-8912-0edd3e50b5d9
wf = resample(sum(sim_data, dims=3)[:,:,1], size(recon));

# ╔═╡ 14a4286e-c8d0-47c0-8cfd-bf1ca3e5f128
show_many(abs.(wf),abs.(ft(wf)).^0.4)

# ╔═╡ f5cd1da8-b927-4c7c-842a-91cbd79d1939
show_many(abs.(recon), abs.(ft(recon)).^0.4)

# ╔═╡ Cell order:
# ╠═0257712c-cb02-4232-a5af-ddfcfb8702ed
# ╟─4ba98557-a551-49d5-ad0f-23a8869d5462
# ╠═4d371c70-f023-46dc-bafa-732f50f20acf
# ╠═ea10443f-6184-4d82-a1e4-519c1a748195
# ╠═b0216ace-c46c-42ee-b35f-8ca894e68a1e
# ╟─6d2ef0c6-6f36-4ab8-8bcd-e5b88daa3eef
# ╠═4c2372be-50bc-11ef-33f1-cbf44b3e823b
# ╠═7306bf9d-4670-4367-8a85-cf0fc1b22ff2
# ╠═792280f3-d803-45c8-aa96-ff329f399e9e
# ╠═cdbe23cb-6622-41fe-98bb-e4d581f64453
# ╠═6b7e4fd9-4ad0-4105-ba18-6b7c63c77528
# ╟─99e542ee-b981-4f2a-88cb-4da222d33e63
# ╠═ce6b9058-e27a-45ed-a216-fb02d06778f2
# ╟─e7656c27-a54d-4679-aa86-e32462456931
# ╠═3be84092-c031-4988-90f0-5bd1e6ac425a
# ╟─0150b9ec-8e40-4e10-bd79-e67795f7d25b
# ╠═ff70b285-f07c-487f-ad36-0989985babe0
# ╠═89034637-b722-4cb9-928b-696fd5283797
# ╟─d832a4c0-2356-4eb9-b758-2a6bc6cbe1b7
# ╟─b890f350-ed74-407c-b390-bc4398a2e9fd
# ╠═459d774a-ba1a-4dc2-bbb7-b0d8e460ac99
# ╟─28e982ee-90d4-42a1-9e5f-d9e4ce171d07
# ╠═ae8ea2d1-68ad-427a-b346-5ca1332f8e37
# ╠═8051f5db-452f-4d45-8102-57892552e45d
# ╟─d4875f35-52f2-4a47-b709-35bb9202cb82
# ╠═f7ac031c-8d9d-4cfa-8912-0edd3e50b5d9
# ╟─a1c2a41b-265b-44d2-9622-520d48f2a9eb
# ╟─7ab1ec5b-84e0-4a2c-9062-4221844e613b
# ╟─a2dd8bb6-62ed-4e01-890f-113c475568d9
# ╠═ca5390fc-6f4f-4b98-b7cd-f61291bbc5f3
# ╟─07b42d3d-374f-403b-b3c0-9af63cf31ea8
# ╠═14a4286e-c8d0-47c0-8cfd-bf1ca3e5f128
# ╟─c00b653d-ede5-45ef-b3b3-4e4527d9f35f
# ╠═f5cd1da8-b927-4c7c-842a-91cbd79d1939
# ╠═3c1c1dd2-ca10-48e1-bfaa-a476b21d5f1a
