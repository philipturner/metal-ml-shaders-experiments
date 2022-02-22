//
//  SGEMVCustom.swift
//  DL4SExperiments
//
//  Created by Philip Turner on 11/24/21.
//

import ARHeadsetUtil
import Metal

extension MMXResources {
    
    func custom_sgemvMethodRaw(commandBuffer: MTLCommandBuffer) {
        let computeEncoder = commandBuffer.makeComputeCommandEncoder()!
        computeEncoder.optLabel = "compute encoder"
        
        computeEncoder.setComputePipelineState(sgemvPipelineState)
        
        struct sgemv_args {
            var matrixRowStride: UInt32
            var inputVectorSize: UInt32
            var transposeMaskStride: UInt32
            
            var inputVectorStride: UInt32
            var outputVectorStride: UInt32
            
            var alpha: Float
            var beta: Float
        }
        
        let rowElementStride = isTranspose ? numRows : numCols
        let rowChunkStride = (rowElementStride + 3) / 4
        let transposeMaskStride = ~7 & (numRows + 7) / 2
        
        let inputVectorStride  = UInt32(~3 & (numCols + 3)) * (inVectorIsBF16  ? 2 : 4)
        let outputVectorStride = UInt32(~3 & (numRows + 3)) * (outVectorIsBF16 ? 2 : 4)
        
        var args = sgemv_args(matrixRowStride: UInt32(rowChunkStride),
                              inputVectorSize: UInt32(numCols),
                              transposeMaskStride: UInt32(transposeMaskStride),
                              
                              inputVectorStride: inputVectorStride,
                              outputVectorStride: outputVectorStride,
                              
                              alpha: 1, beta: 0)
        
        computeEncoder.setBytes(&args, length: MemoryLayout<sgemv_args>.stride, index: 0)
        
        computeEncoder.setBuffer(buffer1, offset: 0, index: 1)
        computeEncoder.setBuffer(buffer2, offset: 0, index: 2)
        computeEncoder.setBuffer(buffer3, offset: 0, index: 3)
        
        if isSparse {
            computeEncoder.setBuffer(buffer4, offset: 0, index: 4)
        }
        
        let threadExecutionWidth = sgemvPipelineState.threadExecutionWidth
        let maxThreadgroupSize = sgemvPipelineState.maxTotalThreadsPerThreadgroup
        
        let needsSharedMemory = usingNonuniformDispatch && (!isTranspose || !supportsSIMDPermute)
        
        if usingNonuniformDispatch {
            func setSharedMemory(groupSize: Int) {
                guard needsSharedMemory else { return }
                
                let sharedMemoryLength = max(16, groupSize / 2 * MemoryLayout<Float>.stride)
                computeEncoder.setThreadgroupMemoryLength(sharedMemoryLength, index: 0)
            }
            
            let roundedChunkStride = roundUpToPowerOf2(rowChunkStride)
            
            if isTranspose {
                var groupWidth = min(roundedChunkStride, 16)
                let groupHeight = min(roundUpToPowerOf2(numCols), min(16, threadExecutionWidth))
                
                if groupWidth * groupHeight > maxThreadgroupSize {
                    groupWidth = maxThreadgroupSize >> groupHeight.trailingZeroBitCount
                }
                
                setSharedMemory(groupSize: groupHeight * groupWidth)
                computeEncoder.dispatchThreads([ groupHeight, roundedChunkStride, numInstances ], threadsPerThreadgroup: [ groupHeight, groupWidth ])
            } else {
                let optimalThreadgroupSize = 64
                var groupWidth = min(roundedChunkStride, optimalThreadgroupSize)
                var simdCount = 1
                
                if supportsSIMDPermute {
                    groupWidth = min(roundedChunkStride, optimalThreadgroupSize)
                    simdCount = max(simdCount, groupWidth >> threadExecutionWidth.trailingZeroBitCount)
                }
                
                groupWidth = min(roundedChunkStride, threadExecutionWidth)
                
                setSharedMemory(groupSize: supportsSIMDPermute ? simdCount : groupWidth)
                computeEncoder.dispatchThreadgroups([ numRows, 1, numInstances ], threadsPerThreadgroup: [ groupWidth, simdCount ])
            }
        } else {
            let numThreads = isTranspose ? rowChunkStride : numRows
            
            if isTranspose {
                computeEncoder.dispatchThreadgroups([ 1, numThreads, numInstances ], threadsPerThreadgroup: [ 1, 1 ])
            } else {
                computeEncoder.dispatchThreadgroups([ numThreads, 1, numInstances ], threadsPerThreadgroup: 1)
            }
        }
        
        computeEncoder.endEncoding()
    }

    func custom_sgemvMethod() -> Int {
        wrapGPUMethod(method: custom_sgemvMethodRaw(commandBuffer:))
    }
    
}
