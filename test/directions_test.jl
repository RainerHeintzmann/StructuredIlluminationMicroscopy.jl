using StructuredIlluminationMicroscopy
using Test

# Directions with different numbers of phases / orders / blocked zero orders

@testset "direction_ranges" begin
    @test direction_ranges(9, 3) == [1:3, 4:6, 7:9]
    @test direction_ranges(11, [3, 5, 3]) == [1:3, 4:8, 9:11]
    @test direction_ranges(5, [5]) == [1:5]
    @test_throws ErrorException direction_ranges(10, [3, 5, 3])
end

@testset "generate_peaks_directions equals generate_peaks for equal directions" begin
    psf = zeros(Float32, 8, 8)
    for (nph, no, blocked) in ((3, 2, false), (3, 3, false), (5, 3, false), (5, 2, true), (3, 3, true))
        kz = blocked ? 0.0 : 0.2           # a blocked zero order is purely lateral (two beams)
        kp, ph, st, oi, op = generate_peaks(3nph, 3, 0.3, no, 0.7, true, kz)
        ph = copy(ph)
        blocked && (ph[:, 2:end] .= mod.(2 .* ph[:, 2:end], 2pi))      # the phase progression doubles
        kp2, ph2, st2, oi2, op2 = generate_peaks_directions(fill(nph, 3), fill(no, 3), fill(blocked, 3), 0.3, 0.7, 0.2)
        @test all(isapprox.(collect.(kp), collect.(kp2); atol=1e-12))
        @test isapprox(ph, ph2; atol=1e-9)
        @test st == st2 && oi == oi2 && op == op2
    end
end

@testset "generate_peaks_directions: 3 / 5 / 3 phases and orders, zero order blocked in the outer directions" begin
    kp, ph, st, oi, op = generate_peaks_directions([3, 5, 3], [3, 5, 3], [true, false, true], 0.0, 0.3, 0.2)
    @test length(kp) == 1 + 2 + 4 + 2                     # shared zero order, then the orders of every direction
    @test size(ph) == size(st) == (11, 9)
    @test all(sum(st; dims=2)[:, 1] .> 0)
    # every image holds the zero order and the peaks of its own direction only
    @test vec(sum(st; dims=2)) == [3, 3, 3, 5, 5, 5, 5, 5, 3, 3, 3]
    # only the medium orders of the not blocked direction are axially modulated
    @test [p[3] for p in kp] == [0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0]
    # the blocked directions' phases advance twice as fast
    @test ph[2, 2] ≈ mod(2 * 2pi * 1 / 3, 2pi)
    @test ph[5, 4] ≈ 2pi * 1 / 5                          # direction 2, phase 2 (q = 1), order 1
    @test_throws ErrorException generate_peaks_directions([3, 5], [3, 5, 3], [false, false], 0.0, 0.3, 0.2)
    @test generate_peaks_directions_param(zeros(Float32, 8, 8), [3, 5, 3], [3, 5, 3], [true, false, true], 0.0, 0.3, 0.2) isa SIMParams
end

@testset "preprocess_sim with directions of different sizes" begin
    dat = 100 .+ 10 .* rand(Float32, 16, 16, 9)
    # equal blocks: the vector form is the same as the number of directions
    a = preprocess_sim(dat; bg=0f0, num_directions=3, correct_drift=false)
    b = preprocess_sim(dat; bg=0f0, num_directions=[3, 3, 3], correct_drift=false)
    @test a == b
    # different numbers of phases: every direction's mean is enforced separately
    dat2 = 100 .+ 10 .* rand(Float32, 16, 16, 11)
    dat2[:, :, 4:8] .*= 2f0                                 # direction 2 twice as bright
    c = preprocess_sim(dat2; bg=0f0, num_directions=[3, 5, 3], correct_drift=false, correct_indiv_brightness=false, damp_edge=false)
    m = [sum(c[:, :, r]) / length(c[:, :, r]) for r in direction_ranges(11, [3, 5, 3])]
    @test maximum(m) / minimum(m) < 1.2
    @test_throws ErrorException preprocess_sim(dat2; bg=0f0, num_directions=[3, 5], correct_drift=false)
end
