module CcodeAPI
# Public API surface
export init!, encode, decode_dfe, decode_isi_bcjr, swap_waveform, sameconv, hardbits

# ── Load local Ccode from source (no Pkg dev required) ────────────────────────
let
    if !isdefined(Main, :Ccode)
        push!(LOAD_PATH, @__DIR__)
        try
            import Ccode   # resolves @__DIR__/Ccode/src/Ccode.jl via its Project.toml
        catch
            push!(LOAD_PATH, joinpath(@__DIR__, "Ccode"))
            import Ccode
        end
    end
end
const Ccode = Main.Ccode  # alias

# ── Config / Lazy init ────────────────────────────────────────────────────────
const _CFG = Base.RefValue{NamedTuple}()   # unassigned Ref; lazily filled by init!()

"Initialize the API and build the trellis (default RSC (5,7), K=3)."
function init!(; K::Integer=3,
                g::NTuple{2,<:Integer}=(0o5, 0o7),
                tail::Bool=false,
                sigma::Real=0.3,
                R::Real=0.5)
    g1, g2 = Int.(g)
    trel   = Ccode.build_rsc_trellis(g1, g2, Int(K))
    _CFG[] = (trel=trel, tail=tail, sigma=Float64(sigma), R=Float64(R))
    return trel
end

@inline function _cfg()
    if !isassigned(_CFG)
        init!()  # lazy default
    end
    return _CFG[]
end

# ── Small utilities exposed ───────────────────────────────────────────────────
"Hard-decision utility (exposes Ccode.hardbits)."
hardbits(x) = Ccode.hardbits(x)

"CAUSAL same-length convolution (kernel first, then crop to input length)."
function sameconv(x::AbstractVector{<:Real}, h::AbstractVector{<:Real})
    y_full = Ccode.fullconv(h, x)
    return Ccode.full_to_same(y_full, h)
end

# ── Encoding ──────────────────────────────────────────────────────────────────
"Convolutional encode info bits using the current trellis."
function encode(u::AbstractVector{<:Integer})
    cfg = _cfg()
    return Ccode.conv_encode(u, cfg.trel; tail=cfg.tail)
end

# ── Waveform swap (noise-preserving LTI remap) ────────────────────────────────
"""
swap_waveform(x_conv, x_ldpc; h=nothing, mode=:best, trim=:same)

Returns (mask::BitVector, out::Vector{Float64}, metric::Float64, choice::Symbol).

If `h === nothing`, `out` is the remapped ±1 sequence.
If `h` is provided, `out` is the remapped **received** sequence, cropped if `trim=:same`.
"""
# Noise-preserving swap and waveform remapping (robust)
function swap(x_conv::AbstractVector{<:Real},
              x_ldpc::AbstractVector{<:Real};
              h=nothing, mode::Symbol=:best, trim::Symbol=:same)

    N = min(length(x_conv), length(x_ldpc))
    @assert N > 0 "swap: input lengths are zero"
    xc = @view x_conv[1:N]
    xl = @view x_ldpc[1:N]

    id_mask   = falses(N)
    glob_mask = trues(N)
    bit_mask  = BitVector(sign.(xc) .!= sign.(xl))  # ensure BitVector

    score(mask::BitVector) = sum(sign.(Float64.(xc) .* (mask .* -2 .+ 1)) .* sign.(xl))

    # ---- initialize defaults so m is always defined ----
    m      = id_mask
    choice = :identity

    if mode === :best
        s_id, s_gl, s_bm = score(id_mask), score(glob_mask), score(bit_mask)
        if s_bm >= s_gl && s_bm >= s_id
            m = bit_mask;  choice = :bitmask
        elseif s_gl >= s_id
            m = glob_mask; choice = :global_flip
        else
            m = id_mask;   choice = :identity
        end
    elseif mode === :identity
        m = id_mask;   choice = :identity
    elseif mode === :global_flip
        m = glob_mask; choice = :global_flip
    elseif mode === :bitmask
        m = bit_mask;  choice = :bitmask
    else
        throw(ArgumentError("swap: unknown mode = $mode"))
    end

    xw  = Float64.(xc) .* (m .* -2 .+ 1)
    met = score(m)

    if h === nothing
        return (m, xw, met, choice)
    else
        y_ldpc_full = Ccode.fullconv(h, xl)
        y_conv_full = y_ldpc_full .+ Ccode.fullconv(h, xw .- xl)
        out = (trim === :same) ? Ccode.full_to_same(y_conv_full, h) : y_conv_full
        return (m, out, met, choice)
    end
end

# (optional alias, if you also expose swap_waveform)
swap_waveform(args...; kwargs...) = swap(args...; kwargs...)


# ── Decoding (DFE + BCJR over LTI-ISI) ────────────────────────────────────────
"""
decode_dfe(y; h, sigma=_cfg().sigma, terminated=_cfg().tail,
           nff=9, nfb=3, dmin=0, dmax=0, lambda=1e-6)

Runs MMSE-DFE front-end and BCJR on the convolutional trellis over an LTI-ISI channel.
`sigma` is **noise std** (this wrapper squares it internally for Ccode).

Returns hard info bits (tail trimmed if `terminated=true` and trellis has memory m>0).
"""
function decode_dfe(y::AbstractVector{<:Real};
                    h::Union{Nothing,AbstractVector{<:Real}}=nothing,
                    sigma::Real=_cfg().sigma,
                    terminated::Bool=_cfg().tail,
                    nff::Int=9, nfb::Int=3,
                    dmin::Int=0, dmax::Int=0,
                    lambda::Real=1e-6)
    cfg  = _cfg()
    trel = cfg.trel
    σ2   = Float64(sigma)^2
    h_eff = h === nothing ? [1.0] : h

    Linfo = Ccode.decode_info_over_isi_lti_dfe(
        y, h_eff, σ2, trel;
        terminated=terminated,
        nff=nff, nfb=nfb,
        dmin=dmin, dmax=dmax,
        lambda=lambda,
    )

    û = Ccode.hardbits(Linfo)
    if terminated && hasproperty(trel, :m)
        m = getfield(trel, :m)
        if m > 0 && length(û) > m
            return û[1:end-m]
        end
    end
    return û
end

# Back-compat alias so old scripts calling `decode(...)` still work
decode(args...; kwargs...) = decode_dfe(args...; kwargs...)

# ── Optional: pure ISI-MAP/BCJR path (wire if available in Ccode) ─────────────
"""
decode_isi_bcjr(y; h, sigma=_cfg().sigma, terminated=_cfg().tail)

Pure ISI-MAP/BCJR (no DFE). Requires Ccode.decode_conv_over_isi_lti to exist.
Returns hard info bits (tail trimmed if applicable).
"""
function decode_isi_bcjr(y::AbstractVector{<:Real};
                         h::AbstractVector{<:Real},
                         sigma::Real=_cfg().sigma,
                         terminated::Bool=_cfg().tail)
    cfg = _cfg()
    if !Base.isdefined(Ccode, :decode_conv_over_isi_lti)
        throw(ArgumentError("Ccode.decode_conv_over_isi_lti not found; add wrapper if/when available."))
    end
    σ2 = Float64(sigma)^2
    Linfo = Ccode.decode_conv_over_isi_lti(y, h, σ2, cfg.trel; terminated=terminated)
    û = Ccode.hardbits(Linfo)
    if terminated && hasproperty(cfg.trel, :m)
        m = getfield(cfg.trel, :m)
        if m > 0 && length(û) > m
            return û[1:end-m]
        end
    end
    return û
end

end # module
