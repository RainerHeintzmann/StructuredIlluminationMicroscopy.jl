
num_directions = 3; num_phases = 3; num_images =  num_phases*num_directions; num_orders = 2
rel_peak = 0.4 # peak position relative to sampling limit on fine grid
# k_peak_pos, peak_phases, peak_strengths, otf_indices, otf_phases = generate_peaks(num_images, num_directions, num_orders, rel_peak / (num_orders-1))
# spf = SIMParams(mypsf, k_peak_pos, peak_phases, peak_strengths, otf_indices, otf_phases);
sp = generate_peaks_param(mypsf, num_images, num_directions, num_orders, rel_peak / (num_orders-1))

msp = randomize_phases(sp, 1.2) # 
# msp = SIMParams(sp)
msp.peak_phases[1,2] = 0.4 # just mess up one phase

sim_data = cat([SIMPattern(msp.mypsf, msp, n, 1) for n=1:num_directions*num_phases]..., dims=3)

rsp = SIMParams(msp)
# rsp.peak_phases = .-(rsp.peak_phases)
@vt ft2d.(test_unmix_real(sim_data, rsp))

# mymat = StructuredIlluminationMicroscopy.weight_matrix(rsp)*3
# mymatb = cat(mymat, conj.(mymat[:,2:end]), dims=2)
# StructuredIlluminationMicroscopy.pinv(mymat)
# StructuredIlluminationMicroscopy.pinv(mymatb)
