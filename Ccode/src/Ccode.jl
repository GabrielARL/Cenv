module Ccode

using LinearAlgebra, Statistics, Printf

# Public exports
export ConvTrellis, build_rsc_trellis, conv_encode,
       llr_from_bpsk_awgn, bcjr_info_llr_from_code_llrs,
       isi_llr_lti, decode_info_over_isi_lti,
       design_mmse_dfe_lti, dfe_symbol_llr_lti, decode_info_over_isi_lti_dfe,
       pretend_swap_fir, fullconv, sameconv, full_to_same, hardbits

# File inclusions (Revise tracks these)
include("conv_utils.jl")
include("trellis.jl")
include("encode.jl")
include("bcjr.jl")
include("isi_lti.jl")
include("dfe_lti.jl")
include("swap.jl")

end # module
