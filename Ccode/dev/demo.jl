#!/usr/bin/env julia
# Demo: LTI ISI with two front-ends + swap
using Random, Statistics, Printf
using Ccode

sigma_from_ebn0_db(ebn0_db, R) = sqrt(1 / (2R * 10.0^(ebn0_db/10)))
ber_from_Linfo(Linfo, u, trel; terminated::Bool=false) =
    mean( (terminated && trel.m>0 ? Ccode.hardbits(Linfo[1:end-trel.m]) : Ccode.hardbits(Linfo)) .!= u )

function main()
    rng = MersenneTwister(0xBEEF)
    trel = Ccode.build_rsc_trellis(0o5, 0o7, 3)
    R, tail, k = 0.5, false, 128
    ebn0 = 2.0
    σ    = sigma_from_ebn0_db(ebn0, R)
    h    = [0.9, 0.4, -0.2]

    @printf("Ccode demo | k=%d | tail=%s | Eb/N0=%.1f dB | σ=%.3f | h=%s\n",
            k, string(tail), ebn0, σ, string(h))

    u = rand(rng, 0:1, k)
    c = Ccode.conv_encode(u, trel; tail=tail)
    x = @. 1 - 2*c

    y_full = Ccode.fullconv(h, x) .+ σ .* randn(rng, length(x)+length(h)-1)
    y_same = Ccode.full_to_same(y_full, h)

    Linfo_bcjr = Ccode.decode_info_over_isi_lti(y_same, h, σ^2, trel; terminated=tail)
    @printf("[ISI-BCJR]  BER = %.4g\n", ber_from_Linfo(Linfo_bcjr, u, trel; terminated=tail))

    Linfo_dfe = Ccode.decode_info_over_isi_lti_dfe(y_same, h, σ^2, trel; terminated=tail, nff=9, nfb=3)
    @printf("[MMSE-DFE]  BER = %.4g\n", ber_from_Linfo(Linfo_dfe, u, trel; terminated=tail))

    c_ldpc = rand(rng, 0:1, length(c))
    x_ldpc = @. 1 - 2*c_ldpc
    y_ldpc_full = Ccode.fullconv(h, x_ldpc) .+ σ .* randn(rng, length(x_ldpc)+length(h)-1)
    y_conv_full = y_ldpc_full .+ Ccode.fullconv(h, x .- x_ldpc)
    res = maximum(abs.((y_ldpc_full .- Ccode.fullconv(h, x_ldpc)) .- (y_conv_full .- Ccode.fullconv(h, x))))
    @printf("[Swap LTI] residual diff = %.3e\n", res)

    y_conv_same = Ccode.full_to_same(y_conv_full, h)
    Linfo_swap  = Ccode.decode_info_over_isi_lti(y_conv_same, h, σ^2, trel; terminated=tail)
    @printf("[Swap LTI] BER after swap = %.4g\n", ber_from_Linfo(Linfo_swap, u, trel; terminated=tail))
end

main()
