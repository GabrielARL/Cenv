# ---------- Code BCJR (info LLRs from code-bit LLRs) ----------
function bcjr_info_llr_from_code_llrs(Lc0::AbstractVector{<:Real},
                                      Lc1::AbstractVector{<:Real},
                                      trel::ConvTrellis; terminated::Bool=false)
    T = length(Lc0); @assert length(Lc1)==T
    ns = trel.ns
    α = fill(-Inf, ns, T+1); β = fill(-Inf, ns, T+1)
    α[1,1] = 0.0; if terminated β[1,end]=0.0 else β[:,end] .= 0.0 end

    Γ0 = Array{Float64}(undef,T,ns); Γ1 = similar(Γ0)
    @inbounds for t in 1:T, s in 1:ns
        (o0,o1) = trel.output_bits[s,1]; c00, c01 = 1-2o0, 1-2o1
        (a0,a1) = trel.output_bits[s,2]; c10, c11 = 1-2a0, 1-2a1
        Γ0[t,s] = 0.5*(float(Lc0[t])*c00 + float(Lc1[t])*c01)
        Γ1[t,s] = 0.5*(float(Lc0[t])*c10 + float(Lc1[t])*c11)
    end
    @inbounds for t in 1:T
        for s in 1:ns
            as = α[s,t]; as==-Inf && continue
            sp0=trel.next_state[s,1]; sp1=trel.next_state[s,2]
            α[sp0,t+1] = jac(α[sp0,t+1], as + Γ0[t,s])
            α[sp1,t+1] = jac(α[sp1,t+1], as + Γ1[t,s])
        end
        c = maximum(@view α[:,t+1]); isfinite(c) && (α[:,t+1] .-= c)
    end
    @inbounds for t in T:-1:1
        for s in 1:ns
            sp0=trel.next_state[s,1]; sp1=trel.next_state[s,2]
            β[s,t] = jac(β[sp0,t+1] + Γ0[t,s], β[sp1,t+1] + Γ1[t,s])
        end
        c = maximum(@view β[:,t]); isfinite(c) && (β[:,t] .-= c)
    end
    Linfo = Vector{Float64}(undef,T)
    @inbounds for t in 1:T
        num = -Inf; den = -Inf
        for s in 1:ns
            sp1=trel.next_state[s,2]; sp0=trel.next_state[s,1]
            num = jac(num, α[s,t] + Γ1[t,s] + β[sp1,t+1])
            den = jac(den, α[s,t] + Γ0[t,s] + β[sp0,t+1])
        end
        Linfo[t] = clampllr(num - den)
    end
    Linfo
end
