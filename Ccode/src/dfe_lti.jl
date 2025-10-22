# ============================================
# src/dfe_lti.jl — DFE front-end (real/BPSK)
# ============================================

# Run DFE and produce symbol LLRs aligned to decisions.
function dfe_symbol_llr_lti(y::AbstractVector{<:Real},
                            h::AbstractVector{<:Real},
                            ff::AbstractVector{<:Real},
                            fb::AbstractVector{<:Real},
                            d::Int, σ2::Real)
    y  = collect(float.(y))
    ff = collect(float.(ff))
    fb = collect(float.(fb))
    T   = length(y)
    nff = length(ff)
    nfb = length(fb)

    # Feed-forward filter output
    z = zeros(Float64, T)
    @inbounds for t in 1:T
        acc = 0.0
        for k in 1:nff
            ti = t - k + 1
            if 1 <= ti <= T
                acc += ff[k] * y[ti]
            end
        end
        z[t] = acc
    end

    # Feedback cancellation with hard decisions
    xhat = zeros(Float64, T)
    @inbounds for t in 1:T
        if nfb > 0
            pc = 0.0
            for i in 1:nfb
                ti = t - i
                if ti >= 1
                    pc += fb[i] * xhat[ti]
                end
            end
            z[t] -= pc
        end
        td = t - d
        if td >= 1
            xhat[t] = (z[td] >= 0 ? +1.0 : -1.0)
        end
    end

    # Effective post-FF noise variance
    σeff2 = σ2 * sum(abs2, ff)

    # Symbol LLRs at decision times (neutral 0.0 where undefined)
    Lx = zeros(Float64, T)
    @inbounds for t in 1:T
        td = t - d
        if 1 <= td <= T
            Lx[t] = clamp((2 / σeff2) * z[td], -50.0, 50.0)
        else
            Lx[t] = 0.0
        end
    end
    return Lx
end

# ---------- MMSE-DFE (LTI) with optimal delay selection ----------
# Chooses delay d in [dmin, dmax] by maximizing p(d)' * inv(R) * p(d),
# where R is the FF autocorr (with noise) and p(d) is the cross-corr vector.
function design_mmse_dfe_lti(h::AbstractVector{<:Real}, σ2::Real;
                             nff::Int=64, nfb::Int=6,
                             dmin::Int=0, dmax::Union{Int,Nothing}=nothing,
                             lambda::Float64=1e-6)
    h = collect(float.(h))
    L = length(h)
    @assert L >= 1
    @assert nff >= 1 && nfb >= 0

    # --- r_yy autocorr [0:maxlag], add noise at lag 0 ---
    maxlag = nff - 1
    ryy = zeros(Float64, maxlag + 1)
    @inbounds for k in 0:maxlag
        s = 0.0
        for i in 1:L
            j = i - k
            if 1 <= j <= L
                s += h[i] * h[j]
            end
        end
        ryy[k + 1] = s
    end
    ryy[1] += σ2

    # Toeplitz R (+ tiny diagonal loading)
    R = zeros(Float64, nff, nff)
    @inbounds for i in 1:nff, j in 1:nff
        R[i, j] = ryy[abs(i - j) + 1]
    end
    @inbounds for i in 1:nff
        R[i, i] += lambda
    end
    Rinvtimes(v) = R \ v

    # Candidate delays
    dmax === nothing && (dmax = min(L - 1, nff - 1))
    @assert 0 <= dmin <= dmax

    best_d = dmin
    best_score = -Inf
    best_ff = nothing

    p = zeros(Float64, nff)
    for d in dmin:dmax
        @inbounds for k in 1:nff
            idx = d - (k - 1) + 1
            p[k] = (1 <= idx <= L) ? h[idx] : 0.0
        end
        ff = Rinvtimes(p)
        score = dot(p, ff)  # p' inv(R) p
        if score > best_score
            best_score = score
            best_d = d
            best_ff = ff
        end
    end
    ff = best_ff::Vector{Float64}
    d  = best_d

    # Combined channel and FB taps
    c  = fullconv(ff, h)
    fb = zeros(Float64, nfb)
    cursor = c[d + 1]
    @inbounds for i in 1:nfb
        tapidx = d + 1 + i
        fb[i] = (tapidx <= length(c)) ? c[tapidx] / cursor : 0.0
    end
    return ff, fb, d
end

# DFE front-end + code BCJR back-end (aligned sys/par split)
function decode_info_over_isi_lti_dfe(y_obs::AbstractVector{<:Real},
                                      h::AbstractVector{<:Real}, σ2::Real,
                                      trel::ConvTrellis;
                                      terminated::Bool=false, nff::Int=7, nfb::Int=3,
                                      dmin::Int=0, dmax::Union{Int,Nothing}=nothing,
                                      lambda::Float64=1e-6)
    @assert iseven(length(y_obs)) "Expect even-length y_obs for rate-1/2."

    # 1) Design DFE and compute symbol LLRs
    ff, fb, d = design_mmse_dfe_lti(h, σ2; nff=nff, nfb=nfb, dmin=dmin, dmax=dmax, lambda=lambda)
    Lx = dfe_symbol_llr_lti(y_obs, h, ff, fb, d, σ2)

    # 2) Drop the first d symbols (invalid decisions before delay)
    start = d + 1
    start = min(start, length(Lx) + 1)
    if start > length(Lx)
        return Float64[]  # nothing usable
    end
    Lxv = @view Lx[start:end]

    # 3) Determine sys/par phase based on delay parity.
    # Each symbol maps to two code bits [sys, par].
    # If d is even ⇒ first bit is SYS ; if d is odd ⇒ first bit is PAR.
    phase = d % 2  # 0 => SYS first, 1 => PAR first

    # Keep only full pairs starting at the right phase.
    len = length(Lxv) - phase
    Tpairs = len ÷ 2
    if Tpairs <= 0
        return Float64[]
    end
    base = phase + 1
    sys_first = base
    par_first = base + 1

    Lc0 = @view Lxv[sys_first:2:(base + 2*Tpairs - 2)]  # systematic
    Lc1 = @view Lxv[par_first:2:(base + 2*Tpairs - 1)]  # parity

    # 4) Code BCJR
    bcjr_info_llr_from_code_llrs(Lc0, Lc1, trel; terminated=terminated)
end
