//
//  MMX.h
//  DL4SExperiments
//
//  Created by Philip Turner on 11/25/21.
//

#ifndef MMX_h
#define MMX_h

#include <metal_stdlib>
using namespace metal;

constant bool is_transpose              [[function_constant(0)]];
constant bool is_sparse                 [[function_constant(1)]];

constant bool matrix_is_bf16            [[function_constant(2)]];
constant bool in_vector_is_bf16         [[function_constant(3)]];
constant bool out_vector_is_bf16        [[function_constant(4)]];

constant bool using_nonuniform_dispatch [[function_constant(5)]];
constant bool supports_simd_permute     [[function_constant(6)]];

#endif /* MMX_h */
