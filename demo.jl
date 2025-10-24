#!/usr/bin/env julia
# demo_api_only_swap_dfe.jl
# API-only demo: Swap (noise-preserving) → DFE+BCJR via CcodeAPI

using Random, Printf, Statistics, Ccode

# Load API as a namespaced alias to avoid symbol conflicts in Main
include("CcodeAPI.jl")
if !isdefined(@__MODULE__, :API)
    API = CcodeAPI           # <- real alias to your API module
else
    @info "Reusing existing API = $(API)"
end



# ----- Helpers -----
sigma_from_ebn0_db(ebn0_db, R) = sqrt(1 / (2R * 10.0^(ebn0_db/10)))

# If API.sameconv is not exposed, provide a 1-liner fallback using swap_waveform(:identity)
if !isdefined(API, :sameconv)
    sameconv(x, h) = API.swap_waveform(x, x; h=h, mode=:identity, trim=:same)[2]
else
    sameconv = API.sameconv
end

# ----- Config (match your demo) -----
rng  = MersenneTwister(0xBEEF)
R, tail, k = 0.5, false, 128
h = [0.9, 0.5, 0.2, 0.1, 0.05, -0.03, 0.02, 0.01, -0.005]
σ = sigma_from_ebn0_db(2.0, R)  # ≈ 0.794

@printf("Swap (DFE) via API-only | k=%d | Eb/N0=%.1f dB | σ=%.3f | h=%s\n",
        k, 2.0, σ, string(h))

# Init API: RSC (5,7) K=3, same tail & sigma as demo
API.init!(K=3, g=(0o5,0o7), tail=tail, sigma=σ, R=R)

# ----- Target conv codeword (the one we want to decode) -----
u = rand(rng, 0:1, k)
c = API.encode(u)
x_conv = @. 1 - 2c              # BPSK (±1)

# ----- "LDPC" stand-in: random bits, different codeword -----
c_ldpc = rand(rng, 0:1, length(c))
x_ldpc = @. 1 - 2c_ldpc

# ----- Receive LDPC waveform through ISI + noise -----
# Use causal same-length conv, then add AWGN
y_ldpc_same = sameconv(x_ldpc, h) .+ σ .* randn(rng, length(x_ldpc))

# ----- Noise-preserving swap → waveform conv would've produced -----
# We need the *noisy* y for LDPC to preserve the exact noise realization.
# API.swap_waveform builds y from clean conv parts; to keep noise, we re-apply the swap formula at symbol rate:
# y_conv_same = y_ldpc_same + (h * x_conv − h * x_ldpc), cropped to "same".
# Since we already have sameconv(), do it explicitly:
y_conv_same = y_ldpc_same .+ (sameconv(x_conv, h) .- sameconv(x_ldpc, h))

# (Optional) score/metric via API to mirror demo prints (not required for correctness)
_, _, met, choice = API.swap_waveform(x_conv, x_ldpc; h=h, mode=:best, trim=:same)
@printf("[Swap LTI] metric = %.3e  choice=%s\n", met, string(choice))

# ----- DFE + BCJR decode via API (lock delay like demo) -----
nff, nfb = 9, 3
û = API.decode(y_conv_same; h=h, sigma=σ,
               terminated=tail, nff=nff, nfb=nfb,
               dmin=0, dmax=0, lambda=1e-6)

# ----- BER against info bits -----
n = min(length(û), length(u))
ber = mean(@view(û[1:n]) .!= @view(u[1:n]))
@printf("[Swap → DFE+BCJR] BER = %.6f (nff=%d, nfb=%d)\n", ber, nff, nfb)
