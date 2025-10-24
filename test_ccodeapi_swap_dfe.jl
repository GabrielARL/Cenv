#!/usr/bin/env julia
# test_ccodeapi_swap_dfe.jl
# Mirror of swap → DFE+BCJR demo, but via CcodeAPI wrappers.

using Random, Printf, Statistics
using Ccode
include("CcodeAPI.jl")
using .CcodeAPI


# --- Helpers (match demo semantics) ---
sigma_from_ebn0_db(ebn0_db, R) = sqrt(1 / (2R * 10.0^(ebn0_db/10)))

# --- Config (short channel, same as demo) ---
rng  = MersenneTwister(0xBEEF)
R, tail, k = 0.5, false, 128
h = [0.9, 0.5, 0.2, 0.1, 0.05, -0.03, 0.02, 0.01, -0.005]
σ = sigma_from_ebn0_db(2.0, R)   # ≈ 0.794

@printf("Swap (DFE) via CcodeAPI | k=%d | Eb/N0=%.1f dB | σ=%.3f | h=%s\n",
        k, 2.0, σ, string(h))

# Init API: RSC (5,7) K=3, same tail & sigma as demo
CcodeAPI.init(K=3, g=(0o5,0o7), tail=tail, sigma=σ, R=R)

# --- Target conv codeword (the one we *want* to decode) ---
u = rand(rng, 0:1, k)
c = CcodeAPI.encode(u)                     # conv encoder via API
x = @. 1 - 2*c                             # BPSK (±1), matches demo (1 - 2c)

# --- "LDPC" stand-in: random bits, different codeword ---
c_ldpc = rand(rng, 0:1, length(c))
x_ldpc = @. 1 - 2*c_ldpc

# --- Receive on the channel (LDPC waveform) ---
y_ldpc_full = Ccode.fullconv(h, x_ldpc) .+ σ .* randn(rng, length(x_ldpc) + length(h) - 1)

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

# --- DFE + BCJR decode via API (lock delay like demo) ---
nff, nfb = 9, 3
û = CcodeAPI.decode(
    y_conv_same; h=h, sigma=σ,
    terminated=tail, nff=nff, nfb=nfb,
    dmin=0, dmax=0, lambda=1e-6
)

# --- BER against info bits ---
n = min(length(û), length(u))
ber = mean(@view(û[1:n]) .!= @view(u[1:n]))
@printf("[Swap → DFE+BCJR] BER = %.6f (nff=%d, nfb=%d)\n", ber, nff, nfb)
