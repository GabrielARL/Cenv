module CcodeAPI

# ── Load local Ccode from source (no Pkg, no UUIDs) ───────────────────────────
let
    # If Ccode isn't already loaded, make the repo root visible to the loader
    if !isdefined(Main, :Ccode)
        # Expecting layout: <this file dir> / Ccode / src / Ccode.jl
        push!(LOAD_PATH, @__DIR__)
        try
            import Ccode   # will resolve @__DIR__/Ccode/src/Ccode.jl
        catch e
            # As a fallback, also try adding the Ccode folder itself
            push!(LOAD_PATH, joinpath(@__DIR__, "Ccode"))
            import Ccode
        end
    end
end
const Ccode = Main.Ccode  # alias for convenience

# ── Config ─────────────────────────────────────────────────────────────────────
const _CFG = Base.RefValue{NamedTuple}()   # unassigned Ref


"Initialize wrapper and build trellis"
function init(; K::Integer=3,
               g::NTuple{2,<:Integer}=(0o5, 0o7),
               tail::Bool=false,
               sigma::Real=0.3,
               R::Real=0.5)

    g1, g2 = Int.(g)                       # <-- coerce to Int
    trel = Ccode.build_rsc_trellis(g1, g2, Int(K))
    _CFG[] = (trel=trel, tail=tail, sigma=Float64(sigma), R=Float64(R))
    return trel
end


@inline function _cfg()
    if !isassigned(_CFG)           # ← don’t read _CFG[] until it’s assigned
        init()                     #   lazily create default cfg
    end
    return _CFG[]
end


"Encode bits using internal Ccode"
function encode(u::AbstractVector{<:Integer})
    cfg = _cfg()
    return Ccode.conv_encode(u, cfg.trel; tail=cfg.tail)
end

"Decode symbols with optional ISI channel using internal DFE+BCJR"
function decode(y::AbstractVector{<:Real};
                h=nothing, terminated::Bool=_cfg().tail,
                nff::Int=9, nfb::Int=3, dmin::Int=0, dmax::Int=0, lambda::Real=1e-6,
                sigma::Real=_cfg().sigma)
    cfg = _cfg()
    trel = cfg.trel
    σ2 = sigma^2
    h_eff = h === nothing ? [1.0] : h

    Linfo = Ccode.decode_info_over_isi_lti_dfe(
        y, h_eff, σ2, trel;
        terminated=terminated, nff=nff, nfb=nfb, dmin=dmin, dmax=dmax, lambda=lambda,
    )

    û = Ccode.hardbits(Linfo)
    if terminated && hasproperty(trel, :m)
        m = getfield(trel, :m)
        if m > 0 && length(û) > m
            û = û[1:end-m]
        end
    end
    return û
end

"Noise-preserving swap and waveform remapping"
function swap(x_conv::AbstractVector{<:Real},
              x_ldpc::AbstractVector{<:Real};
              h=nothing, mode::Symbol=:best, trim::Symbol=:same)

    N = min(length(x_conv), length(x_ldpc))
    xc = @view x_conv[1:N]
    xl = @view x_ldpc[1:N]

    id_mask   = falses(N)
    glob_mask = trues(N)
    bit_mask  = sign.(xc) .!= sign.(xl)

    score(mask) = sum(sign.(Float64.(xc) .* (mask .* -2 .+ 1)) .* sign.(xl))

    local m::BitVector, choice::Symbol
    if mode == :best
        s_id = score(id_mask); s_gl = score(glob_mask); s_bm = score(bit_mask)
        if s_bm >= s_gl && s_bm >= s_id
            m = bit_mask; choice = :bitmask
        elseif s_gl >= s_id
            m = glob_mask; choice = :global_flip
        else
            m = id_mask; choice = :identity
        end
    elseif mode == :identity
        m = id_mask; choice = :identity
    elseif mode == :global_flip
        m = glob_mask; choice = :global_flip
    elseif mode == :bitmask
        m = bit_mask; choice = :bitmask
    else
        throw(ArgumentError("unknown mode=$mode"))
    end

    xw = Float64.(xc) .* (m .* -2 .+ 1)
    met = score(m)

    if h === nothing
        return (m, xw, met, choice)
    else
        y_ldpc_full = Ccode.fullconv(h, xl)
        y_conv_full = y_ldpc_full .+ Ccode.fullconv(h, xw .- xl)
        out = (trim == :same) ? Ccode.full_to_same(y_conv_full, h) : y_conv_full
        return (m, out, met, choice)
    end
end

end # module
