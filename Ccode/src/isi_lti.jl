# ---------- ISI (LTI) symbol-MAP trellis ----------
@inline _nx(s::Int, b::Int, m::Int) = m==0 ? 1 : (((s-1)>>1) | (b<<(m-1))) + 1

function isi_llr_lti(y::AbstractVector{<:Real}, h::AbstractVector{<:Real}, σ2::Real)
    T = length(y); L = length(h); m = L - 1; ns = 1 << max(m, 0)
    α = fill(-Inf, ns, T+1); β = fill(-Inf, ns, T+1)
    α[:,1] .= 0.0; β[:,T+1] .= 0.0

    prevx = zeros(Float64, ns, max(m, 0))
    @inbounds for s in 0:ns-1, l in 1:m
        prevx[s+1,l] = 1.0 - 2.0 * ((s >> (m - l)) & 1)
    end

    inv2σ2 = 0.5 / float(σ2)
    Γ0 = Array{Float64}(undef, T, ns)
    Γ1 = similar(Γ0)

    @inbounds for t in 1:T
        up = min(m, t-1)
        for s in 1:ns
            sum_prev = 0.0
            @inbounds for l in 1:up
                sum_prev += float(h[l+1]) * prevx[s,l]
            end
            h1 = float(h[1])
            e0 = float(y[t]) - ( h1*(+1.0) + sum_prev )
            e1 = float(y[t]) - ( h1*(-1.0) + sum_prev )
            Γ0[t,s] = -(e0*e0)*inv2σ2
            Γ1[t,s] = -(e1*e1)*inv2σ2
        end
    end

    @inbounds for t in 1:T
        for s in 1:ns
            a=α[s,t]; a==-Inf && continue
            sp0=_nx(s,0,m); sp1=_nx(s,1,m)
            α[sp0,t+1]=jac(α[sp0,t+1], a+Γ0[t,s])
            α[sp1,t+1]=jac(α[sp1,t+1], a+Γ1[t,s])
        end
        c=maximum(@view α[:,t+1]); isfinite(c) && (α[:,t+1] .-= c)
    end
    @inbounds for t in T:-1:1
        for s in 1:ns
            sp0=_nx(s,0,m); sp1=_nx(s,1,m)
            β[s,t]=jac(β[sp0,t+1]+Γ0[t,s], β[sp1,t+1]+Γ1[t,s])
        end
        c=maximum(@view β[:,t]); isfinite(c) && (β[:,t] .-= c)
    end

    Lx=Vector{Float64}(undef,T)
    @inbounds for t in 1:T
        num=-Inf; den=-Inf
        for s in 1:ns
            sp0=_nx(s,0,m); sp1=_nx(s,1,m)
            num=jac(num, α[s,t]+Γ0[t,s]+β[sp0,t+1])  # x_t=+1
            den=jac(den, α[s,t]+Γ1[t,s]+β[sp1,t+1])  # x_t=-1
        end
        Lx[t]=clampllr(num-den)
    end
    Lx
end

function decode_info_over_isi_lti(y_obs::AbstractVector{<:Real},
                                  h::AbstractVector{<:Real}, σ2::Real,
                                  trel::ConvTrellis; terminated::Bool=false)
    @assert iseven(length(y_obs))
    Lx = isi_llr_lti(y_obs, h, σ2)
    Lc0=@view Lx[1:2:end]; Lc1=@view Lx[2:2:end]
    bcjr_info_llr_from_code_llrs(Lc0,Lc1,trel; terminated=terminated)
end
