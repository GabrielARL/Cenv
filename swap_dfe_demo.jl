#!/usr/bin/env julia
using Random, Printf
using Ccode, Statistics

# --- Helpers ---
sigma_from_ebn0_db(ebn0_db, R) = sqrt(1 / (2R * 10.0^(ebn0_db/10)))
ber_from_Linfo(Linfo, u, trel; terminated=false) = begin
    û = terminated && trel.m>0 ? Ccode.hardbits(Linfo[1:end-trel.m]) : Ccode.hardbits(Linfo)
    n = min(length(û), length(u))
    n == 0 && return 0.5
    mean(@view(û[1:n]) .!= @view(u[1:n]))
end

# --- Config (short channel example; you can swap in long ones too) ---
rng  = MersenneTwister(0xBEEF)
trel = Ccode.build_rsc_trellis(0o5, 0o7, 3)
R, tail, k = 0.5, false, 128
h = [0.9, 0.5, 0.2, 0.1, 0.05, -0.03, 0.02, 0.01, -0.005]
σ = sigma_from_ebn0_db(2.0, R)

@printf("Swap (DFE) demo | k=%d | Eb/N0=%.1f dB | σ=%.3f | h=%s\n",
        k, 2.0, σ, string(h))

# --- Target conv codeword (the one we *want* to decode) ---
u = rand(rng, 0:1, k)
c = Ccode.conv_encode(u, trel; tail=tail)
x = @. 1 - 2*c

# --- "LDPC" stand-in: random bits, different codeword ---
c_ldpc = rand(rng, 0:1, length(c))
x_ldpc = @. 1 - 2*c_ldpc

# --- Receive on the channel (LDPC waveform) ---
y_ldpc_full = Ccode.fullconv(h, x_ldpc) .+ σ .* randn(rng, length(x_ldpc)+length(h)-1)

# --- Noise-preserving swap to what the conv codeword would have produced ---
# y_conv_full = y_ldpc_full + fullconv(h, x - x_ldpc)
y_conv_full = y_ldpc_full .+ Ccode.fullconv(h, x .- x_ldpc)

# Residual sanity (should be ~0): (noise term must match)
res = maximum(abs.(
    (y_ldpc_full .- Ccode.fullconv(h, x_ldpc)) .- (y_conv_full .- Ccode.fullconv(h, x))
))
@printf("[Swap LTI] residual diff = %.3e\n", res)

# --- Trim to symbol-rate (same length as x) ---
y_conv_same = Ccode.full_to_same(y_conv_full, h)

# --- DFE + BCJR decode (you can tune nff/nfb; for short h we also lock d=0) ---
nff, nfb = 9, 3
Linfo_swap_dfe = Ccode.decode_info_over_isi_lti_dfe(
    y_conv_same, h, σ^2, trel;
    terminated=tail,
    nff=nff, nfb=nfb,
    dmin=0, dmax=0,   # lock delay for this short channel; remove/relax for long channels
    lambda=1e-6
)
ber_swap_dfe = ber_from_Linfo(Linfo_swap_dfe, u, trel; terminated=tail)
@printf("[Swap → DFE+BCJR] BER = %.6f (nff=%d, nfb=%d)\n", ber_swap_dfe, nff, nfb)
