//
//  RandomTape.h
//  DL4SExperiments
//
//  Created by Philip Turner on 11/24/21.
//

#ifndef RandomTape_h
#define RandomTape_h

#include <metal_stdlib>
using namespace metal;

// Expand this in the future to extract values for dropout

class RandomTape {
    device void *raw_bits;
public:
    RandomTape(device void *raw_bits)
    {
        this->raw_bits = raw_bits;
    }
    
    // Extract 8 bits
    
    half extract_float_8(uint index)
    {
        ushort random_bits = reinterpret_cast<device uchar*>(raw_bits)[int(index)];
        return half(random_bits) / half(ulong(1) << 8);
    }
    
    half2 extract_float_8x2(uint index)
    {
        uchar2 random_bits = reinterpret_cast<device uchar2*>(raw_bits)[int(index)];
        return half2(random_bits) / half2(ulong(1) << 8);
    }
    
    // Extract 16 bits
    
    float extract_float_16(uint index)
    {
        ushort random_bits = reinterpret_cast<device ushort*>(raw_bits)[int(index)];
        return float(random_bits) / float(ulong(1) << 16);
    }
    
    float2 extract_float_16x2(uint index)
    {
        ushort2 random_bits = reinterpret_cast<device ushort2*>(raw_bits)[int(index)];
        return float2(random_bits) / float2(ulong(1) << 16);
    }
    
    // Extract 32 bits
    
    float extract_float_32(uint index)
    {
        uint random_bits = reinterpret_cast<device uint*>(raw_bits)[int(index)];
        return float(random_bits) / float(ulong(1) << 32);
    }
    
    float2 extract_float_32x2(uint index)
    {
        uint2 random_bits = reinterpret_cast<device uint2*>(raw_bits)[int(index)];
        return float2(random_bits) / float2(ulong(1) << 32);
    }
};

#endif /* RandomTape_h */
