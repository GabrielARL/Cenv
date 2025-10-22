# ---------- Utils ----------
@inline jac(a::Float64, b::Float64) = (a==-Inf ? b : b==-Inf ? a : begin
    m = ifelse(a>b,a,b); m + log1p(exp(-abs(a-b)))
end)

@inline clampllr(x; L=50.0) = clamp(float(x), -L, L)
hardbits(L::AbstractVector) = Int.(L .> 0)

# ---------- Convolution ----------
function fullconv(h::AbstractVector{<:Real}, x::AbstractVector{<:Real})
    L, N = length(h), length(x)
    y = zeros(Float64, N+L-1)
    @inbounds for i in 1:N
        xi = float(x[i])
        @simd for k in 1:L
            y[i+k-1] += float(h[k])*xi
        end
    end
    y
end

sameconv(h::AbstractVector{<:Real}, x::AbstractVector{<:Real}) = fullconv(h,x)[1:length(x)]
full_to_same(y_full::AbstractVector{<:Real}, h::AbstractVector{<:Real}) = y_full[1:length(y_full)-length(h)+1]

llr_from_bpsk_awgn(y::AbstractVector{<:Real}, σ2::Real) = (2/float(σ2)) .* float.(y)
