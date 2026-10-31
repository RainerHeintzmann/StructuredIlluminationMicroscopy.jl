using Test
using StructuredIlluminationMicroscopy

# `recon_sim_finalize!` (no allocation of arrays: the shifts are folded into modulations) equals the allocating
# computation, for the final filter or none, with or without the final fftshift, for even and odd sizes
@testset "recon_sim_finalize!" begin
    nd, nph, no = 3, 3, 2
    nimg = nd * nph
    mkdat(sz) = Float32[1f0 + 0.5f0 * cos(2π * 0.625 * (y - sz[2] / 2) + d * 2π / nph) + 0.1f0 * sin(0.3f0 * x)
                        for x in 1:sz[1], y in 1:sz[2], z in 1:sz[3], d in 1:nimg]
    mkpsf(sz) = Float32[exp(-((x - sz[1] / 2)^2 / 128) - ((y - sz[2] / 2)^2 / 128) - ((z - sz[3] / 2)^2 / 72))
                        for x in 1:sz[1], y in 1:sz[2], z in 1:sz[3]]
    for (sz, ups, fast) in (((32, 32, 4), 2, true), ((32, 32, 1), 2, true), ((30, 34, 6), 2, true), ((33, 31, 5), 1, false))
        sim = preprocess_sim(mkdat(sz); bg=0f0, num_directions=nd, correct_drift=false)
        sp = generate_peaks_param(mkpsf(sz), nimg, nd, 0.0, no, 0.9, true, 0.0)
        prep = recon_sim_prepare(sim, sp, ReconParams(; double_use=false, upsample_factor=ups); use_final_filter=true)
        res, bsz = recon_sim_orders(sim, prep, sp)           # = prep.result_rft
        @test !Base.mightalias(res, prep.result_rft_tmp)     # the scratch array is separate
        res_ref = copy(res)
        res0 = res
        for shift in (true, false), ff in (prep.final_filter, ones(Bool, 1))
            ref = StructuredIlluminationMicroscopy._finalize_alloc(prep, res0, bsz, ff, shift)
            dst = similar(res0, real(eltype(res0)), bsz...)
            out = recon_sim_finalize!(dst, prep, res0, bsz; final_filter=ff, shift=shift)
            @test out === dst
            @test maximum(abs.(out .- ref)) <= 2e-5 * maximum(abs.(ref))
            @test recon_sim_finalize(prep, res0, bsz; final_filter=ff, shift=shift) ≈ ref rtol=1e-4
        end
        @test res == res_ref                                  # the spectrum is not modified
        dst = similar(res0, real(eltype(res0)), bsz...)
        recon_sim_finalize!(dst, prep, res0, bsz)
        fast && @test (@allocated recon_sim_finalize!(dst, prep, res0, bsz)) < 10_000   # no array is allocated
        # recon_sim(inplace=true) returns the preallocated result, equal to the standard one
        r = recon_sim(sim, prep, sp)
        ri = recon_sim(sim, prep, sp; inplace=true)
        @test ri === prep.result
        @test maximum(abs.(ri .- r)) <= 1e-4 * maximum(abs.(r))
    end
end
