//
//  Random.metal
//  DL4SExperiments
//
//  Created by Philip Turner on 11/24/21.
//

#include <metal_stdlib>
#include "../BFloat16/BFloat16.h"
#include "PhiloxRNG.h"
#include "RandomDistribution.h"
#include "RandomTape.h"
using namespace metal;

constant bool supports_long [[function_constant(0)]];

typedef struct {
    uint4 counter_base;
    uint2 key;
} philox_args;

kernel void generateRandomTape(constant philox_args &args [[buffer(0)]],
                               device   uint4       *tape [[buffer(1)]],
                               
                               uint id [[thread_position_in_grid]])
{
    uint4 counter = PhiloxRNG::incrementCounter(args.counter_base, id);
    tape[int(id)] = PhiloxRNG::random(counter, args.key, supports_long);
}



constant bool   generating_bf16   [[function_constant(1)]];
constant ushort distribution_type [[function_constant(2)]];

typedef struct {
    float param_a;
    float param_b;
} random_dist_args;

kernel void generateRandomDistribution(constant random_dist_args &args      [[buffer(0)]],
                                       device   void             *tape_bits [[buffer(1)]],
                                       device   void             *output    [[buffer(2)]],
                                       
                                       uint id [[thread_position_in_grid]])
{
    RandomTape tape(tape_bits);
    using RandDist = RandomDistribution;
    
    switch (RandDist::DistributionType(distribution_type))
    {
        case RandDist::uniform:
        {
            if (generating_bf16)
            {
                float tape_val = tape.extract_float_16(id);
                float uniform_val = RandDist::get_uniform(tape_val, args.param_a, args.param_b);
                
                reinterpret_cast<device bfloat16*>(output)[int(id)] = uniform_val;
            }
            else
            {
                float tape_val = tape.extract_float_32(id);
                float uniform_val = RandDist::get_uniform(tape_val, args.param_a, args.param_b);
                
                reinterpret_cast<device float*>(output)[int(id)] = uniform_val;
            }
            
            break;
        }
        case RandomDistribution::normal2:
        {
            if (generating_bf16)
            {
                float2 tape_val = tape.extract_float_16x2(id);
                float2 normal_val = RandDist::get_normal2<half>(tape_val, args.param_a, args.param_b);
                
                reinterpret_cast<device bfloat16_2*>(output)[int(id)] = normal_val;
            }
            else
            {
                float2 tape_val = tape.extract_float_32x2(id);
                float2 normal_val = RandDist::get_normal2<float>(tape_val, args.param_a, args.param_b);
                
                reinterpret_cast<device float2*>(output)[int(id)] = normal_val;
            }
            
            break;
        }
        case RandomDistribution::bernoulli:
        {
            float tape_val = tape.extract_float_16(id);
            
            if (generating_bf16)
            {
                bfloat16 bernoulli_val = RandDist::get_bernoulli<bfloat16>(tape_val, args.param_a);
                reinterpret_cast<device bfloat16*>(output)[int(id)] = bernoulli_val;
            }
            else
            {
                float bernoulli_val = RandDist::get_bernoulli<float>(tape_val, args.param_a);
                reinterpret_cast<device float*>(output)[int(id)] = bernoulli_val;
            }
            
            break;
        }
    }
}
