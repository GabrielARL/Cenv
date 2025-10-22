# ---------- RSC encoder (rate-1/2) ----------
function conv_encode(u::AbstractVector{<:Integer}, trel::ConvTrellis; tail::Bool=true)
    s=1; m=trel.m; T=length(u)+(tail ? m : 0); y=Vector{Int}(undef,2T); j=1
    @inbounds for b in u
        (o0,o1)=trel.output_bits[s,Int(b)+1]; y[j]=o0; y[j+1]=o1; j+=2
        s=trel.next_state[s,Int(b)+1]
    end
    if tail
        for _ in 1:m
            (o0,o1)=trel.output_bits[s,1]; y[j]=o0; y[j+1]=o1; j+=2
            s=trel.next_state[s,1]
        end
    end
    y
end
