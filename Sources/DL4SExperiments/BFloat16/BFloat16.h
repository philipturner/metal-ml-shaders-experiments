//
//  BFloat16.h
//  DL4SExperiments
//
//  Created by Philip Turner on 11/16/21.
//

#ifndef BFloat16_h
#define BFloat16_h

#include <metal_stdlib>
using namespace metal;

// bfloat16

class bfloat16 {
    ushort storage;
    
    friend class bfloat16_2;
    friend class bfloat16_3;
    friend class bfloat16_4;
    
public:
    bfloat16(float float_val)
    {
        storage = as_type<ushort2>(float_val)[1];
    }
    
    float unpacked() const
    {
        ushort2 vector(0, storage);
        
        return as_type<float>(vector);
    }
    
    static float read_any(device void *raw_pointer, uint index, bool is_bf16)
    {
        if (is_bf16)
        {
            auto temp = reinterpret_cast<device bfloat16*>(raw_pointer)[int(index)];
            return temp.unpacked();
        }
        else
        {
            return reinterpret_cast<device float*>(raw_pointer)[int(index)];
        }
    }
    
//    static float read_any(threadgroup void *raw_pointer, uint index, bool is_bf16)
//    {
//        if (is_bf16)
//        {
//            auto temp = reinterpret_cast<threadgroup bfloat16*>(raw_pointer)[int(index)];
//            return temp.unpacked();
//        }
//        else
//        {
//            return reinterpret_cast<threadgroup float*>(raw_pointer)[int(index)];
//        }
//    }
    
    static void write_any(device void *raw_pointer, uint index, float value, bool is_bf16)
    {
        if (is_bf16)
        {
            reinterpret_cast<device bfloat16*>(raw_pointer)[int(index)] = { value };
        }
        else
        {
            reinterpret_cast<device float*>(raw_pointer)[int(index)] = value;
        }
    }
    
//    static void write_any(threadgroup void *raw_pointer, uint index, float value, bool is_bf16)
//    {
//        if (is_bf16)
//        {
//            reinterpret_cast<threadgroup bfloat16*>(raw_pointer)[int(index)] = { value };
//        }
//        else
//        {
//            reinterpret_cast<threadgroup float*>(raw_pointer)[int(index)] = value;
//        }
//    }
};

// bfloat16_2

class bfloat16_2 {
    ushort2 storage;
    
public:
    bfloat16_2(float2 float_val)
    {
        storage = {
            bfloat16(float_val[0]).storage,
            bfloat16(float_val[1]).storage
        };
    }
    
    float2 unpacked() const
    {
        ushort2 vector1(0, storage[0]);
        ushort2 vector2(0, storage[1]);
        
        return {
            as_type<float>(vector1),
            as_type<float>(vector2)
        };
    }
};

// bfloat16_3

class bfloat16_3 {
    ushort3 storage;
    
public:
    bfloat16_3(float3 float_val)
    {
        storage = {
            bfloat16(float_val[0]).storage,
            bfloat16(float_val[1]).storage,
            bfloat16(float_val[2]).storage
        };
    }
    
    float3 unpacked() const
    {
        ushort2 vector1(0, storage[0]);
        ushort2 vector2(0, storage[1]);
        ushort2 vector3(0, storage[2]);
        
        return {
            as_type<float>(vector1),
            as_type<float>(vector2),
            as_type<float>(vector3)
        };
    }
};

// bfloat16_4

class bfloat16_4 {
    ushort4 storage;
    
public:
    bfloat16_4(float4 float_val)
    {
        storage = {
            bfloat16(float_val[0]).storage,
            bfloat16(float_val[1]).storage,
            bfloat16(float_val[2]).storage,
            bfloat16(float_val[3]).storage
        };
    }
    
    float4 unpacked() const
    {
        ushort2 vector1(0, storage[0]);
        ushort2 vector2(0, storage[1]);
        ushort2 vector3(0, storage[2]);
        ushort2 vector4(0, storage[3]);
        
        return {
            as_type<float>(vector1),
            as_type<float>(vector2),
            as_type<float>(vector3),
            as_type<float>(vector4)
        };
    }
    
    static float4 read_any(device void *raw_pointer, uint index, bool is_bf16, ushort mask_bits = 0b1111)
    {
        auto mask = bool4(mask_bits & ushort4(1, 2, 4, 8));
        
        if (is_bf16)
        {
            auto temp = reinterpret_cast<device ushort4*>(raw_pointer)[int(index)];
            temp = select(0, temp, mask);
            
            return reinterpret_cast<thread bfloat16_4&>(temp).unpacked();
        }
        else
        {
            auto temp = reinterpret_cast<device float4*>(raw_pointer)[int(index)];
            temp = select(0, temp, mask);
            
            return temp;
        }
    }
    
    static void write_any(device void *raw_pointer, uint index, float4 value, bool is_bf16)
    {
        if (is_bf16)
        {
            reinterpret_cast<device bfloat16_4*>(raw_pointer)[int(index)] = { value };
        }
        else
        {
            reinterpret_cast<device float4*>(raw_pointer)[int(index)] = value;
        }
    }
};

#endif /* BFloat16_h */
