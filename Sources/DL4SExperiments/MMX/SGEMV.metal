//
//  SGEMV.metal
//  DL4SExperiments
//
//  Created by Philip Turner on 11/13/21.
//

#include <metal_stdlib>
#include "../BFloat16/BFloat16.h"
#include "MMX.h"
using namespace metal;

typedef struct {
    uint matrix_row_stride; // # of 4-element chunks each row spans in memory
    uint input_vector_size;
    uint transpose_mask_stride; // # of bytes in each row of the swizzled transpose mask - CHANGE to: read from a non-swizzled index matrix, but will not be a transpose multiplication since stuff is transposed beforehand
    
    // all strides should be aligned to 4-element boundaries
    uint input_vector_stride; // # of bytes separating each instance of the input vector
    uint output_vector_stride; // # of bytes separating each instance of the output vector
    
    float alpha;
    float beta;
} sgemv_args;

inline float4 read_matrix_chunk(device void *raw_matrix, device uchar *masks, uint matrix_index, uint mask_index)
{
    if (is_sparse)
    {
        ushort mask = masks[mask_index / 2];
        mask = select(mask % 16, mask / 16, mask_index % 2);
        
        return bfloat16_4::read_any(raw_matrix, matrix_index, matrix_is_bf16, mask);
    }
    else
    {
        return bfloat16_4::read_any(raw_matrix, matrix_index, matrix_is_bf16);
    }
}

template <ushort loop_size>
inline void reduce_sum(threadgroup float *shared_mem, thread void *raw_sum, ushort3 thread_id, ushort3 tg_size)
{
    auto sum = reinterpret_cast<thread float*>(raw_sum);
    
    for (ushort i = tg_size.x / 2; i > 0; i /= 2)
    {
        for (ushort j = 0; j < loop_size; ++j)
        {
            if (!supports_simd_permute && thread_id.x >= i)
            {
                simdgroup_barrier(mem_flags::mem_threadgroup);
                
                shared_mem[thread_id.x - i] = sum[j];
                
                simdgroup_barrier(mem_flags::mem_threadgroup);
            }
            
            if (thread_id.x + i < tg_size.x)
            {
                sum[j] += supports_simd_permute ? simd_shuffle_down(sum[j], i) : shared_mem[thread_id.x];
            }
        }
    }
    
    if (!is_transpose && supports_simd_permute &&
        tg_size.y > 1 && thread_id.x == 0)
    {
        for (ushort i = tg_size.y / 2; i > 0; i /= 2)
        {
            if (thread_id.y >= i)
            {
                shared_mem[thread_id.y - i] = sum[0];
            }
            
            threadgroup_barrier(mem_flags::mem_threadgroup);
            
            if (thread_id.y < i)
            {
                sum[0] += shared_mem[thread_id.y];
            }
            
            if (i > 1)
            {
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
        }
    }
}



constant bool needs_shared_memory = using_nonuniform_dispatch && (!is_transpose || !supports_simd_permute);

kernel void sgemv(constant sgemv_args &args          [[buffer(0)]],
                  device   void       *matrix        [[buffer(1)]],
                  device   void       *input_vector  [[buffer(2)]],
                  device   void       *output_vector [[buffer(3)]],
                  device   uchar      *input_masks   [[buffer(4), function_constant(is_sparse)]], // replace this with an entirely different mechanism
                  
                  threadgroup float *raw_shared_mem [[threadgroup(0), function_constant(needs_shared_memory)]],
                  
                  ushort3 thread_id    [[thread_position_in_threadgroup]],
                  ushort  thread_index [[thread_index_in_threadgroup]],
                  ushort3 tg_size      [[threads_per_threadgroup]],
                  uint3   tid          [[thread_position_in_grid]],
                  uint3   tgid         [[threadgroup_position_in_grid]])
{
    uint matrix_offset = is_transpose ? tid.y : (tgid.x * args.matrix_row_stride);
    device void *raw_matrix = (device uchar*)matrix + matrix_offset * (matrix_is_bf16 ? 8 : 16);
    
    device void *input  = (device uchar*)input_vector  + args.input_vector_stride  * tid.z;
    device void *output = (device uchar*)output_vector + args.output_vector_stride * tid.z;
    
    device uchar *selected_masks = input_masks;
    
    if (!is_transpose)
    {
        if (is_sparse)
        {
            selected_masks += matrix_offset / 2;
        }
        
        uint i_end_upper = (args.input_vector_size + 3) / 4;
        uint i_end_lower =  args.input_vector_size / 4;
        
        // Accumulate row
        
        float sum = 0;
        
        for (uint i = thread_index; i < i_end_upper; i += tg_size.x * tg_size.y)
        {
            float4 retrieved_matrix = read_matrix_chunk(raw_matrix, selected_masks, i, i);
            float4 retrieved_vector = bfloat16_4::read_any(input, i, in_vector_is_bf16);
            
            if (i == i_end_lower)
            {
                for (ushort j = args.input_vector_size % 4; j < 4; ++j)
                {
                    retrieved_matrix[j] = 0;
                    retrieved_vector[j] = 0;
                }
            }
            
            sum += dot(retrieved_matrix, retrieved_vector);
        }
        
        if (using_nonuniform_dispatch)
        {
            reduce_sum<1>(raw_shared_mem, &sum, thread_id, tg_size);
        }
        
        if (thread_index == 0)
        {
            if (args.alpha != 1) { sum *= args.alpha; }
            if (args.beta  != 0) { sum += args.beta * bfloat16::read_any(output, tgid.x, out_vector_is_bf16); }
            
            bfloat16::write_any(output, tgid.x, sum, out_vector_is_bf16);
        }
    }
    else
    {
        if (is_sparse)
        {
            selected_masks += tid.y * args.transpose_mask_stride;
        }
        
        uint matrix_index;
        uint loop_matrix_stride;
        
        if (tg_size.x == 1)
        {
            matrix_index = 0;
            loop_matrix_stride = args.matrix_row_stride;
        }
        else
        {
            matrix_index = thread_id.x * args.matrix_row_stride;
            loop_matrix_stride = tg_size.x * args.matrix_row_stride;
        }
        
        // Accumulate rows
        
        float4 sum = 0;
        
        for (uint i = thread_id.x; i < args.input_vector_size; i += tg_size.x)
        {
            float4 retrieved_matrix = read_matrix_chunk(raw_matrix, selected_masks, matrix_index, i);
            matrix_index += loop_matrix_stride;
            
            float retrieved_vector = bfloat16::read_any(input, i, in_vector_is_bf16);
            sum += retrieved_matrix * retrieved_vector;
        }
        
        if (using_nonuniform_dispatch && tg_size.x > 1)
        {
            auto shared_mem = raw_shared_mem + thread_id.y * (tg_size.x / 2);
            reduce_sum<4>(shared_mem, &sum, thread_id, tg_size);
        }
        
        if (thread_id.x == 0)
        {
            if (args.alpha != 1) { sum *= args.alpha; }
            if (args.beta  != 0) { sum += args.beta * bfloat16_4::read_any(output, tid.y, out_vector_is_bf16); }
            
            bfloat16_4::write_any(output, tid.y, sum, out_vector_is_bf16);
        }
    }
}
