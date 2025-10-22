# ---------- Noise-preserving swap (LTI) ----------
function pretend_swap_fir(y::AbstractVector{<:Real},
                          cA::AbstractVector{<:Integer}, cB::AbstractVector{<:Integer},
                          h::AbstractVector{<:Real}; use_full::Bool=true)
    @assert length(cA)==length(cB)
    xA = @. 1 - 2*cA; xB = @. 1 - 2*cB; Δx = xB .- xA
    y_corr = fullconv(h,Δx)
    yB = y .+ (use_full ? y_corr : y_corr[1:length(y)])
    yB
end
