#!/usr/bin/env julia
using Test
using Random
using Ccode

# --- Helpers ---
sigma_from_ebn0_db(ebn0_db, R) = sqrt(1 / (2R * 10.0^(ebn0_db/10)))

# Length-safe BER: trims Linfo (and tail if terminated) and compares over overlap only
ber_from_Linfo(Linfo, u, trel; terminated::Bool=false) = begin
    û_full = terminated && trel.m > 0 ? Ccode.hardbits(Linfo[1:end - trel.m]) :
                                        Ccode.hardbits(Linfo)
    n = min(length(û_full), length(u))
    n == 0 && return 0.5
    mean(@view(û_full[1:n]) .!= @view(u[1:n]))
end

# --- Deterministic setup ---
rng  = MersenneTwister(0xBEEF)
trel = Ccode.build_rsc_trellis(0o5, 0o7, 3)  # (5,7), K=3, rate-1/2
R, tail, k = 0.5, false, 128
h = [0.9, 0.5, 0.2, 0.1, 0.05, -0.03, 0.02, 0.01, -0.005]
σ = sigma_from_ebn0_db(2.0, R)               # Eb/N0 = 2 dB

# --- Encode ---
u = rand(rng, 0:1, k)
c = Ccode.conv_encode(u, trel; tail=tail)    # length = 2k (sys,par interleaved)
x = @. 1 - 2*c                               # BPSK

# --- Channel ---
y_full = Ccode.fullconv(h, x) .+ σ .* randn(rng, length(x) + length(h) - 1)
y_same = Ccode.full_to_same(y_full, h)       # same-length observation

@testset "Ccode smoke tests" begin
    # (1) ISI-BCJR path
    Linfo_bcjr = Ccode.decode_info_over_isi_lti(y_same, h, σ^2, trel; terminated=tail)
    ber_bcjr   = ber_from_Linfo(Linfo_bcjr, u, trel; terminated=tail)
    @test 0.0 <= ber_bcjr <= 0.20

    # (2) DFE+BCJR path — lock delay d=0 for this tiny channel (keeps sys/par phase correct)
    Linfo_dfe = Ccode.decode_info_over_isi_lti_dfe(
        y_same, h, σ^2, trel;
        terminated=tail,
        nff=9, nfb=3,
        dmin=0, dmax=0,     # <- force d = 0
        lambda=1e-6
    )
    ber_dfe = ber_from_Linfo(Linfo_dfe, u, trel; terminated=tail)
    @test 0.0 <= ber_dfe <= 0.35

    # (3) Noise-preserving swap sanity (random "LDPC" bits -> conv codeword)
    c_ldpc = rand(rng, 0:1, length(c))
    x_ldpc = @. 1 - 2*c_ldpc
    y_ldpc_full = Ccode.fullconv(h, x_ldpc) .+ σ .* randn(rng, length(x_ldpc) + length(h) - 1)
    y_conv_full = y_ldpc_full .+ Ccode.fullconv(h, x .- x_ldpc)   # swap to conv, noise preserved

    # Residual of noise terms should be ~0
    res = maximum(abs.((y_ldpc_full .- Ccode.fullconv(h, x_ldpc)) .- (y_conv_full .- Ccode.fullconv(h, x))))
    @test res ≤ 1e-12

    # Decode swapped observation (conv target)
    y_conv_same = Ccode.full_to_same(y_conv_full, h)
    Linfo_swap  = Ccode.decode_info_over_isi_lti(y_conv_same, h, σ^2, trel; terminated=tail)
    ber_swap    = ber_from_Linfo(Linfo_swap, u, trel; terminated=tail)
    @info "[ISI-BCJR]"  BER=ber_bcjr
    @info "[DFE+BCJR]" BER=ber_dfe
    @info "[Swap→BCJR]" BER=ber_swap
    @test 0.0 <= ber_swap <= 0.30
end
