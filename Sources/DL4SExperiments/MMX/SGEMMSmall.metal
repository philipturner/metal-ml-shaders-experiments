//
//  SGEMMSmall.metal
//  DL4SExperiments
//
//  Created by Philip Turner on 11/25/21.
//

#include <metal_stdlib>
#include "../BFloat16/BFloat16.h"
#include "MMX.h"
using namespace metal;

constant ushort tile_size [[function_constant(7)]]; // must be between 4 and 32
constant bool is_aligned  [[function_constant(8)]];
constant bool needs_matrix_tile = !supports_simd_permute || is_transpose;

typedef struct {
    uint matrix_row_stride; // # of 4-element chunks each row spans in memory
    uint input_vector_size;
    uint output_vector_size;
    
    // all strides should be aligned to 4-element boundaries
    uint input_vector_stride; // # of bytes separating each instance of the input vector
    uint output_vector_stride; // # of bytes separating each instance of the output vector
    uint num_model_instances;
    
    float alpha;
    float beta;
} sgemm_args;

inline void mask_value(ushort num_valid_lanes, thread float4 &value)
{
    for (ushort i = num_valid_lanes; i < 4; ++i)
    {
        value[num_valid_lanes] = 0;
    }
}

inline void shared_memory_barrier(ushort simd_size)
{
    if (tile_size * tile_size <= simd_size)
    {
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    else
    {
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}



kernel void sgemm_small(constant sgemm_args &args          [[buffer(0)]],
                        device   void       *matrix        [[buffer(1)]],
                        device   void       *input_vector  [[buffer(2)]],
                        device   void       *output_vector [[buffer(3)]],
                        // sparse never occurs in combination with transpose
                        
                        threadgroup float4 *matrix_tile [[threadgroup(0), function_constant(needs_matrix_tile)]],
                        threadgroup float4 *vector_tile [[threadgroup(1)]],
                        
                        ushort  simd_size    [[thread_execution_width]],
                        ushort2 thread_id    [[thread_position_in_threadgroup]],
                        ushort  lane_id      [[thread_index_in_simdgroup]],
                        ushort  thread_index [[thread_index_in_threadgroup]],
                        uint2   tid          [[thread_position_in_grid]],
                        uint2   tgid         [[threadgroup_position_in_grid]])
{
    uint col;
    uint col_4th;
    ushort2 swizzled_id;
    
    uint matrix_offset;
    uint matrix_device_stride;
    
    if (!is_transpose)
    {
        // dispatch X -> row -> horizontal
        // dispatch Y -> col -> vertical
        
        col = tid.y;
        
        matrix_offset = thread_id.x + tid.y * args.matrix_row_stride;
        matrix_device_stride = tile_size;
    }
    else
    {
        // dispatch X -> row -> vertical
        // dispatcy Y -> col -> horizontal
        
        swizzled_id.x = thread_id.x * 4;
        swizzled_id.x += thread_id.y / (tile_size / 4);
        swizzled_id.y  = thread_id.y % (tile_size / 4);
        
        col_4th = tgid.y * (tile_size / 4) + swizzled_id.y;
        
        matrix_offset = col_4th + swizzled_id.x * args.matrix_row_stride;
        matrix_device_stride = tile_size * 4 * args.matrix_row_stride;
    }
    
    ushort model_instance = tgid.x * tile_size + thread_id.y;
    device void *input  = (device uchar*)input_vector  + args.input_vector_stride  * model_instance;
    device void *output = (device uchar*)output_vector + args.output_vector_stride * tid.x;
    
    uint row_4th_end_hi = (args.input_vector_size + 3) / 4;
    uint row_4th_end_lo =  args.input_vector_size / 4;
    
    uint col_4th_end_hi = (args.output_vector_size + 3) / 4;
    uint col_4th_end_lo =  args.output_vector_size / 4;
    
    // Accumulate sum
    
    float sum = 0;
    
    for (uint row_4th = 0; row_4th < row_4th_end_hi; row_4th += tile_size)
    {
        // Read matrix and vector from device memory
        
        float4 cached_matrix;
        float4 cached_vector;
        
        uint thread_row_4th = row_4th + thread_id.x;;
        uint row = row_4th * 4 + swizzled_id.x;
        
        if (is_aligned ||
            (!is_transpose ? (thread_row_4th < row_4th_end_hi && col < args.output_vector_size)
                           :        (col_4th < col_4th_end_hi && row < args.input_vector_size)))
        {
            cached_matrix = bfloat16_4::read_any(matrix, matrix_offset, matrix_is_bf16);
            matrix_offset += matrix_device_stride;
            
            if (!is_aligned &&
                (!is_transpose ? thread_row_4th == row_4th_end_lo
                               :        col_4th == col_4th_end_lo))
            {
                uint vec_size = !is_transpose ? args.input_vector_size : args.output_vector_size;
                mask_value(vec_size % 4, cached_matrix);
            }
        }
        else
        {
            cached_matrix = 0;
        }
        
        if (is_aligned || (thread_row_4th < row_4th_end_hi && model_instance < args.num_model_instances))
        {
            cached_vector = bfloat16_4::read_any(input, row_4th, in_vector_is_bf16);
            
            if (!is_aligned && (thread_row_4th == row_4th_end_lo))
            {
                mask_value(args.input_vector_size % 4, cached_vector);
            }
        }
        else
        {
            cached_vector = 0;
        }
        
        // Transpose matrix
        
        shared_memory_barrier(simd_size);
        
        if (is_transpose)
        {
            ushort tile_index = swizzled_id.y * tile_size * 16 + swizzled_id.x;
            
            for (ushort j = 0; j < 4; ++j)
            {
                ((threadgroup float*)matrix_tile)[tile_index] = cached_matrix[j];
                tile_index += tile_size * 4;
            }
        }
        else if (needs_matrix_tile)
        {
            matrix_tile[thread_index] = cached_matrix;
        }
        
        vector_tile[thread_index] = cached_vector;
        shared_memory_barrier(simd_size);
        
        if (is_transpose && supports_simd_permute)
        {
            cached_matrix = matrix_tile[thread_index];
        }
        
        // Multiply matrix by vector
        
        ushort lane_mask = ~(tile_size - 1);
        ushort vector_index = thread_id.x * tile_size;
        ushort j_start;
        
        if (supports_simd_permute)
        {
            if (simd_size == tile_size)
            {
                j_start = 0;
            }
            else
            {
                j_start = lane_id & lane_mask;
            }
        }
        else
        {
            j_start = thread_index & lane_mask;
        }
        
        for (ushort j = j_start; j < j_start + tile_size; ++j)
        {
            float4 retrieved_matrix;
            float4 retrieved_vector = vector_tile[vector_index];
            
            if (supports_simd_permute)
            {
                retrieved_matrix = simd_broadcast(cached_matrix, j);
            }
            else
            {
                retrieved_matrix = matrix_tile[j];
            }
            
            sum += dot(retrieved_matrix, retrieved_vector);
            vector_index += 1;
        }
    }
    
    if (is_aligned || (tid.x < args.num_model_instances && tid.y < args.output_vector_size))
    {
        if (args.alpha != 1) { sum *= args.alpha; }
        if (args.beta  != 0) { sum += args.beta * bfloat16::read_any(output, tid.y, out_vector_is_bf16); }
        
        bfloat16::write_any(output, tid.y, sum, out_vector_is_bf16);
    }
}
