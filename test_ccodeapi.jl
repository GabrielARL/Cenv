#!/usr/bin/env julia
# test_ccodeapi.jl
# Smoke test for CcodeAPI: encode → ISI channel → three decoders:
#   [ISI-BCJR], [DFE+BCJR], [Swap→BCJR]
#
# Uses your current CcodeAPI.decode for the DFE path and auto-detects optional others.

using Random, Printf, LinearAlgebra

# --- Load packages/APIs ---
include("CcodeAPI.jl")
using .CcodeAPI
using Ccode  # to reuse fullconv/full_to_same helpers

# ---------------- Utilities ----------------
"""Return the first function that exists in CcodeAPI from a list of candidate names."""
function _resolve_api(candidates::Vector{String})
    for name in candidates
        if Base.isdefined(CcodeAPI, Symbol(name)) &&
           getfield(CcodeAPI, Symbol(name)) isa Function
            return getfield(CcodeAPI, Symbol(name)), name
        end
    end
    return nothing, ""
end

"""Same-length convolution (like 'sameconv') using Ccode helpers."""
function sameconv(x::AbstractVector{<:Real}, h::AbstractVector{<:Real})
    yfull = Ccode.fullconv(collect(float.(x)), collect(float.(h)))
    return Ccode.full_to_same(yfull, h)
end

"""Compute BER between 0/1 vectors"""
ber(a::AbstractVector{<:Integer}, b::AbstractVector{<:Integer}) = sum(a .!= b) / length(a)

# Coerce decoder outputs to 0/1 info bits
function coerce_bits(v)
    T = eltype(v)
    if T <: AbstractFloat
        return Int.(v .> 0)
    elseif T <: Integer
        return Int.(v)
    else
        return Int.(v .> 0)
    end
end

function ber_against_info(û, K, u)
    ûi = coerce_bits(û)
    if length(ûi) != K
        ûi = ûi[1:K]  # naive slice for smoke test if codebits returned
    end
    return ber(ûi, u)
end

# Call decoders with flexible arg orders
function call_decoder(fun::Function, y, h, σ)
    try
        return fun(y, h, σ)
    catch
        try
            return fun(y, σ, h)
        catch
            return fun(y, σ)
        end
    end
end

# ---------------- Resolve API symbols ----------------
# Encoder
enc_fun, _ = _resolve_api(["conv_encode", "encode_conv", "encode"])
enc_fun === nothing && (enc_fun = CcodeAPI.encode)

# Optional decoders; we'll fall back for DFE below
isi_fun,  _ = _resolve_api(["decode_isi_bcjr", "isi_bcjr_decode", "bcjr_isi", "decode_over_isi_bcjr"])
dfe_fun,  _ = _resolve_api(["decode_dfe", "decode_dfe_bcjr", "dfe_then_bcjr", "mmse_dfe_decode"])
swap_fun, _ = _resolve_api(["decode_swap_bcjr", "swap_then_bcjr", "pretend_swap_bcjr", "swap_bcjr"])

# Fallback: use your existing API for DFE path
if dfe_fun === nothing
    dfe_fun = (y,h,σ) -> CcodeAPI.decode(y; h=h, sigma=σ)
end

# ---------------- Parameters ----------------
seed   = 0xC0FFEE
rng    = MersenneTwister(seed)

K      = 256            # info bits
σ      = 0.25           # AWGN std
h_raw  = [0.9, 0.35, -0.2]     # simple LTI ISI
h      = h_raw ./ norm(h_raw)

@info "CcodeAPI test" seed=seed K=K σ=σ h=h

# ---------------- Generate & encode ----------------
u = rand(rng, 0:1, K)
c = enc_fun(u)                                 # convolutional code bits
@printf("Encoded length = %d\n", length(c))

# BPSK, ISI channel + AWGN
x       = @. 2c - 1.0
y_clean = sameconv(x, h)
y       = y_clean .+ σ .* randn(rng, length(y_clean))

# ---------------- Decoders ----------------

# --- [ISI-BCJR] ---
if isi_fun !== nothing
    û_isi  = call_decoder(isi_fun, y, h, σ)
    ber_isi = ber_against_info(û_isi, K, u)
    @printf("[ISI-BCJR]\nBER = %.6f\n", ber_isi)
else
    @printf("[ISI-BCJR]\nSKIP (no %s found)\n", "decode_isi_bcjr/isi_bcjr_decode/bcjr_isi/decode_over_isi_bcjr")
end

# --- [DFE+BCJR] --- (always runs via fallback at least)
û_dfe  = call_decoder(dfe_fun, y, h, σ)
ber_dfe = ber_against_info(û_dfe, K, u)
@printf("[DFE+BCJR]\nBER = %.6f\n", ber_dfe)

# --- [Swap→BCJR] ---
if swap_fun !== nothing
    û_swap  = call_decoder(swap_fun, y, h, σ)
    ber_swap = ber_against_info(û_swap, K, u)
    @printf("[Swap→BCJR]\nBER = %.6f\n", ber_swap)
else
    @printf("[Swap→BCJR]\nSKIP (no %s found)\n", "decode_swap_bcjr/swap_then_bcjr/pretend_swap_bcjr/swap_bcjr")
end

println()
@info "Done ✅"
