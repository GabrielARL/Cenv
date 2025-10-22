#!/usr/bin/env julia
using Random, Statistics, Printf, Test
using Ccode

# Helpers
sigma_from_ebn0_db(ebn0_db, R) = sqrt(1 / (2R * 10.0^(ebn0_db/10)))
ber_from_Linfo(Linfo, u, trel; terminated::Bool=false) = begin
    # Turn info-LLRs into hard decisions (trim tail if terminated)
    û_full = terminated && trel.m > 0 ? Ccode.hardbits(Linfo[1:end-trel.m]) :
                                        Ccode.hardbits(Linfo)
    # Compare only over the overlapping prefix (DFE may return fewer bits)
    n = min(length(û_full), length(u))
    n == 0 && return 0.5   # degenerate fallback
    mean(@view(û_full[1:n]) .!= @view(u[1:n]))
end


# Build a 100-tap real channel with a decaying mainlobe + small tails
function make_long_h(; L=100, main=15, decay=0.8, rng=MersenneTwister(42))
    h = zeros(Float64, L)
    # main lobe
    for i in 1:main
        h[i] = decay^(i-1)
    end
    # add small random postcursors
    for i in (main+1):L
        h[i] = 0.08 * randn(rng)
    end
    # normalize energy
    h ./= sqrt(sum(abs2, h) + eps())
    return h
end

# ---- Config ----
rng = MersenneTwister(0xBEEF)
trel = Ccode.build_rsc_trellis(0o5, 0o7, 3)  # (5,7), K=3
R, tail, k = 0.5, false, 1024
h = make_long_h(L=100, main=15, decay=0.85; rng=rng)
EbN0dB = 4.0
σ = sigma_from_ebn0_db(EbN0dB, R)

@printf("DFE long-ISI demo | k=%d | L=%d | Eb/N0=%.1f dB | σ=%.3f\n", k, length(h), EbN0dB, σ)

# Bits → conv encode
u = rand(rng, 0:1, k)
c = Ccode.conv_encode(u, trel; tail=tail)
x = @. 1 - 2*c

# Channel (define y BEFORE calling DFE)
y = Ccode.fullconv(h, x)
y = y[1:length(x)] .+ σ .* randn(rng, length(x))

# ---- DFE front-end + code BCJR back-end ----
nff, nfb = 128, 8
ff, fb, d = Ccode.design_mmse_dfe_lti(h, σ^2; nff=nff, nfb=nfb, dmin=0, dmax=40, lambda=1e-5)
@printf("[DFE-opt] nff=%d nfb=%d delay=%d  ||ff||²=%.3f  max|fb|=%.3f\n",
        nff, nfb, d, sum(abs2, ff), maximum(abs.(fb)))

Linfo_dfe = Ccode.decode_info_over_isi_lti_dfe(y, h, σ^2, trel;
    terminated=tail, nff=nff, nfb=nfb, dmin=0, dmax=40, lambda=1e-5)
ber_dfe   = ber_from_Linfo(Linfo_dfe, u, trel; terminated=tail)
@printf("[DFE+BCJR] BER = %.4g\n", ber_dfe)

@test 0.0 <= ber_dfe <= 0.28  # adjust bound as needed for your channel/SNR

