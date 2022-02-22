//
//  SGEMMCustom.swift
//  DL4SExperiments
//
//  Created by Philip Turner on 11/26/21.
//

import ARHeadsetUtil
import Metal

extension MMXResources {
    
    func custom_sgemmSmallMethodRaw(commandBuffer: MTLCommandBuffer) {
        precondition(numInstances > 1, "Use sgemv when running 1 or 2 model instances")
        
        let computeEncoder = commandBuffer.makeComputeCommandEncoder()!
        computeEncoder.optLabel = "compute encoder"
        computeEncoder.setComputePipelineState(sgemmPipelineState)
        
        struct sgemm_args {
            var matrixRowStride: UInt32
            var inputVectorSize: UInt32
            var outputVectorSize: UInt32
            
            var inputVectorStride: UInt32
            var outputVectorStride: UInt32
            var numModelInstances: UInt32
            
            var alpha: Float
            var beta: Float
        }
        
        let rowElementStride = isTranspose ? numRows : numCols
        let rowChunkStride = (rowElementStride + 3) / 4
        
        let inputVectorStride  = UInt32(~3 & (numCols + 3)) * (inVectorIsBF16  ? 2 : 4)
        let outputVectorStride = UInt32(~3 & (numRows + 3)) * (outVectorIsBF16 ? 2 : 4)
        
        var args = sgemm_args(matrixRowStride: UInt32(rowChunkStride),
                              inputVectorSize: UInt32(numCols),
                              outputVectorSize: UInt32(numRows),
                              
                              inputVectorStride: inputVectorStride,
                              outputVectorStride: outputVectorStride,
                              numModelInstances: UInt32(numInstances),
                              
                              alpha: 1, beta: 0)
        
        computeEncoder.setBytes(&args, length: MemoryLayout<sgemm_args>.stride, index: 0)
        
        computeEncoder.setBuffer(buffer1, offset: 0, index: 1)
        computeEncoder.setBuffer(buffer2, offset: 0, index: 2)
        computeEncoder.setBuffer(buffer3, offset: 0, index: 3)
        
        // Allocate shared memory
        
        let needsMatrixTile = !supportsSIMDPermute || isTranspose
        
        func setSharedMemory(isBF16: Bool, index: Int) {
            let sharedMemoryLength = max(16, tileSize * tileSize * MemoryLayout<simd_float4>.stride)
            computeEncoder.setThreadgroupMemoryLength(sharedMemoryLength, index: index)
        }
        
        if needsMatrixTile { setSharedMemory(isBF16: matrixIsBF16, index: 0) }
        setSharedMemory(isBF16: inVectorIsBF16, index: 1)
        
        // Dispatch threads
        
        var numGroupsX: Int
        var numGroupsY: Int
        
        if isAligned {
            numGroupsX = numInstances >> tileSize.trailingZeroBitCount
            numGroupsY = numRows      >> tileSize.trailingZeroBitCount
        } else {
            numGroupsX = (numInstances + tileSize - 1) >> tileSize.trailingZeroBitCount
            numGroupsY = (numRows      + tileSize - 1) >> tileSize.trailingZeroBitCount
        }
        
        numGroupsX = max(1, numGroupsX)
        numGroupsY = max(1, numGroupsY)
        
        computeEncoder.dispatchThreadgroups([ numGroupsX, numGroupsY ], threadsPerThreadgroup: [ tileSize, tileSize ])
        computeEncoder.endEncoding()
    }
    
    func custom_sgemmMethod() -> Int {
        wrapGPUMethod(method: custom_sgemmSmallMethodRaw(commandBuffer:))
    }
    
}
