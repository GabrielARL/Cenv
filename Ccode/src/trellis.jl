# ---------- Trellis ----------
struct ConvTrellis
    K::Int; m::Int; ns::Int
    G::NTuple{2,Int}
    next_state::Matrix{Int}
    output_bits::Matrix{NTuple{2,Int}}
end

function build_rsc_trellis(num::Integer, den::Integer, K::Integer)
    K  = Int(K); @assert K≥2; m = K-1; ns = 1<<m
    mask = (1<<K)-1; num&=mask; den&=mask; @assert (den & 0x1)==1
    nsM  = zeros(Int, ns, 2); out = Matrix{NTuple{2,Int}}(undef, ns, 2)
    @inline getv(s,i) = (s >> (m - i)) & 1
    @inbounds for s in 0:ns-1, u in 0:1
        fb = 0;  for i in 1:m; ((den>>i)&1)==1 && (fb ⊻= getv(s,i)); end
        v  = u ⊻ fb
        p  = ((num&1)==1) ? v : 0
        for i in 1:m; ((num>>i)&1)==1 && (p ⊻= getv(s,i)); end
        sp = (s >> 1) | (v << (m-1))
        nsM[s+1,u+1]  = sp+1
        out[s+1,u+1]  = (u, p)
    end
    ConvTrellis(K,m,ns,(num,den),nsM,out)
end
