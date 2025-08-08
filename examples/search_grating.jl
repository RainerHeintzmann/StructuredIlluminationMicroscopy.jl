# a 1-color example generating SLM-based sim gratings optimized to fit trough holes in a mask 

using StructuredIlluminationMicroscopy

path = raw".\\"
# path = raw"examples\\"
start = -20; # Start pixel of the scanning range ( foral l4 dimensions )
stop = 80 ; # End pixel of the scanning range )
num_dir = 3 ; # Number directions
num_phase = 3 ; # Number phases
wl = [488];
# wl = [488, 561, 638]; # A list with all wavelength to be scanned
bfp_fill = 66 ; # 80.6   or 66

# Back focal plane filling (referning to M, and NA below ) : If a grating period in pixesl is given that value will be ignored . The Value referes to the first wavelengths in the wl list . For all other values the grating periods will be adapted in order to ensure the same diffraction angle
err_p = 0.05#0 . 5 ; # Max error of the grating period in percent
err_a = 1 ; # standard # max error of the angle in degrees
fixed_angle=nothing ; # define starting angle -> all possible starting angles will be scanned if None
pxs = 20 ; # Pixel size of SLM ( only important if bfp fill is used)
w_gauss = 0.5; # size of the gaussian laser spot on the SLM in cm (important for estimating unwanted orders )
h =0.3; # hole diameter of the fourier mask
dim_slm = (600, 800); # python backwards definition
dim_slm = (800, 600); # julia definition
f = 500; # focal length of the collimating lens behind SLM
NA = 1.4 ; #NA of Objective (only important if bfp fill is used )
M = 152.7 ; #DeMagnification of the grating (only important if bfp fill is used)
per = nothing ; #Desired grating period in pixels
fixed_angle_set = nothing; 
# If a value is given ( c.f. next line) the vector that defines the angle of the grating is fixed to this value ! By that you can accurately define the grating angle e.g. for NL_SIM (for ensuring the colinearity of the gratings ) or for generating exactly perpendicular gratings (e.g. for 2D Grating SIM)
#fixed_angle_set = [ 5, 34 ] ;
phase_check_method = 2 ;
"""
PhaseStepMethod 1 or 2 :
There is a discrepancy between the
Phasestepmethod in Ronny's Paper and in the original Matlab code
With 1 or 2 you can choose either one of them!
The flag is used in function "check phase steps"
1.) Method as descriped in the paper
+ Accurate with respect to the phase step check (nip.sim.testgrat)
+ As it has been published it SHOULD (!!!) give accurate phase steps
- A very large scanning range might be necessary as it is a very harsh criterium
2.) Method as in the old Matlabcode from Ronny
+ A looser criterium as 1 and thus requires smaller scanning range
- However Igenerally did not encounter any problems yet, it is not one houndret percent clear what happened here and if it always gives correct phase steps .
In case you use this : Check the grating safterwards
"""

optimize = true; # Minimize Unwanted o rde r s ( should be t rue )
opt_grat_sum = false; 
# do you want to optimize the grating sums? In this case for eachpotential grating set , the num phase gratings will be computed, filtered and summed up. The ratio between the summed peak-to-peak and the ptp_value for one grating is below 5% the parameter set will be excepted. Otherwise it will be expelled.
#generation for the opt_grating_sum> how to shift the phases ? 1:shift phases between 0 and pi, 2:shift hte phases between 0 and 2 pi;

# GENERATE THE GRATINGS: the parameters will be saved and returned
# Also refere to sim.creategratingparamfile?
pl = create_grating_param_file(path; start=start, stop=stop,
    num_dir=num_dir,num_phase =num_phase,wavelength=wl,error_period=
    err_p,error_angle=err_a,fixed_angle=fixed_angle,px_size=pxs,w_gauss=w_gauss,h=h,dim_slm=dim_slm,f=f,NA=NA,magnification=M,
    BFP_filling=bfp_fill,period=per,fixed_angle_set=fixed_angle_set,
    PhaseCheckMethod=phase_check_method,optimize_for_unwanted_orders=optimize,opt_grating_sum=opt_grat_sum,generation=1)

# myphase = 1
# grating = StructuredIlluminationMicroscopy.generate_grating(pl[1].opt_para, myphase, num_phase; dim_slm=dim_slm, method="binary", blaze=0, periods=1, binary_threshold=0)

save_grat_folder = path * "gratings"  
to_read = path * "para_4.032_3phases_3Dir.txt"  
grating = create_grating(to_read, save_grat_folder; dim_slm=dim_slm, circle_aperture_radius=-1, name_tag="", 
                        test_grating_shift=false, bitdepth=1, form="png", val="max", method="binary", version=2,
                        blaze_vec=nothing, func=1);

ps = pl[1]
start_dir = 1.0
period = StructuredIlluminationMicroscopy.get_first_period(ps)
wanted_mask, unwanted_mask = StructuredIlluminationMicroscopy.generate_mask(3, start_dir, dim_slm, h, wl[1], period, pxs; f=f)
@ve wanted_mask unwanted_mask

# Now we can create the grating

q = StructuredIlluminationMicroscopy.generate_grating(ps.para_list, 0, num_phase; dim_slm=dim_slm)
@vt wanted_mask abs.(ft2d(reshape(q, (size(q)[1:2]..., 1, 1, size(q)[3]))))

# Now we can find the grating
grating_found = find_grating(grating, para, dim_slm, bfp_fill=bfp_fill, wavelength=wl[1], phase_nr=1, blaze=0.0);

# Now we can check the grating
check_phase_steps(grating_found, para, dim_slm, bfp_fill=bfp_fill, wavelength=wl[1], phase_nr=1, blaze=0.0);

# Now we can optimize the grating sum