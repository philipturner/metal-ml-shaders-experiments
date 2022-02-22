//
//  PhiloxRNG.h
//  DL4SExperiments
//
//  Created by Philip Turner on 11/24/21.
//

#ifndef PhiloxRNG_h
#define PhiloxRNG_h

#include <metal_stdlib>
using namespace metal;

/**
 Outputs the exact same stream of random bits as Swift for TensorFlow's Philox random number generator.
 
 Source: https://github.com/tensorflow/swift-apis/blob/main/Sources/Tensor/Random.swift
 */
class PhiloxRNG {
private:
    static void bump(thread uint2 &key)
    {
        constexpr uint bump_constant_hi = 0x9E3779B9;
        constexpr uint bump_constant_lo = 0xBB67AE85;
        
        key = { key[0] + bump_constant_lo, key[1] + bump_constant_hi };
    }
    
    // TODO: - after debugging, test whether this can be optionally accelerated by 64-bit arithmetic
    
    static void round(thread uint4 &counter, uint2 key, bool supports_long)
    {
        constexpr uint round_constant_0 = 0xD2511F53;
        constexpr uint round_constant_1 = 0xCD9E8D57;
        
        if (supports_long)
        {
            ulong val0 = ulong(round_constant_0) * ulong(counter[3]);
            ulong val1 = ulong(round_constant_1) * ulong(counter[1]);
            
            uint hi0 = as_type<uint2>(val0)[1];
            uint lo0 = as_type<uint2>(val0)[0];
            
            uint hi1 = as_type<uint2>(val1)[1];
            uint lo1 = as_type<uint2>(val1)[0];
            
            counter = {
                lo0, hi0 ^ counter[0] ^ key[0],
                lo1, hi1 ^ counter[2] ^ key[1]
            };
        }
        else
        {
            uint hi0 = mulhi(round_constant_0, counter[3]);
            uint lo0 = round_constant_0 * counter[3];
            
            uint hi1 = mulhi(round_constant_1, counter[1]);
            uint lo1 = round_constant_1 * counter[1];
            
            counter = {
                lo0, hi0 ^ counter[0] ^ key[0],
                lo1, hi1 ^ counter[2] ^ key[1]
            };
        }
    }
    
public:
    static uint4 random(uint4 counter, uint2 key, bool supports_long)
    {
        // Round 1
        
        uint4 counter_copy = counter;
        uint2 key_copy = key;
        
        round(counter_copy, key_copy, supports_long);
        
        // Rounds 2 - 10
        
        for (ushort i = 1; i < 10; ++i)
        {
            bump(key_copy);
            round(counter_copy, key_copy, supports_long);
        }
        
        return {
            counter_copy[2], counter_copy[3],
            counter_copy[0], counter_copy[1]
        };
    }
    
    static uint4 incrementCounter(uint4 base, uint offset)
    {
        uint4 output = base;
        
        // Guard against overflow
        
        if (__UINT32_MAX__ - output[0] <= offset)
        {
            for (ushort i = 1; i < 4; ++i)
            {
                if (output[i] < __UINT32_MAX__)
                {
                    output[i] += 1;
                    break;
                }

                output[i] = 0;
            }
        }
        
        output[0] += offset;
        
        return output;
    }
};

#endif /* PhiloxRNG_h */
