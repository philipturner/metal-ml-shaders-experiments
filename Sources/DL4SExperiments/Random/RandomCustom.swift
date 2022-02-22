//
//  RandomCustom.swift
//  DL4SExperiments
//
//  Created by Philip Turner on 11/24/21.
//

import ARHeadsetUtil
import Metal

extension RandomResources {
    
    func custom_randomMethodRaw(commandBuffer: MTLCommandBuffer) {
        let computeEncoder = commandBuffer.makeComputeCommandEncoder()!
        computeEncoder.optLabel = "compute encoder"
        
        struct philox_args {
            var counterBase: simd_uint4
            var key: simd_uint2
        }
        
        var args = philox_args(counterBase: .zero,
                               key: unsafeBitCast(key, to: simd_uint2.self))
        
        computeEncoder.setComputePipelineState(tapePipelineState)
        computeEncoder.setBytes(&args, length: MemoryLayout<philox_args>.stride, index: 0)
        
        computeEncoder.setBuffer(randomNumberBuffer, offset: 0, index: 1)
        computeEncoder.dispatchThreads([ (randomNumberSize + 3) / 4 ], threadsPerThreadgroup: 1)
        
        computeEncoder.endEncoding()
    }
    
    func custom_randomMethod() -> Int {
        wrapGPUMethod(method: custom_randomMethodRaw(commandBuffer:))
    }
    
    // make a custom_randomDistribution method
    //
    // debug it to make sure it produces the correct distribution, don't profile it
    
    func custom_distributionMethodRaw(commandBuffer: MTLCommandBuffer) {
        let computeEncoder = commandBuffer.makeComputeCommandEncoder()!
        computeEncoder.optLabel = "compute encoder"
        
        struct random_dist_args {
            var paramA: Float
            var paramB: Float
        }
        
        var args = random_dist_args(paramA: distributionType == .uniform ? (paramB - paramA) : (paramA ?? 0),
                                    paramB: distributionType == .uniform ? paramA : (paramB ?? 0))
        
        print("args", args)
        
        // make dist output = random number size, that way no overshoot
        
        computeEncoder.setComputePipelineState(distributionPipelineState)
        computeEncoder.setBytes(&args, length: MemoryLayout<random_dist_args>.stride, index: 0)
        
        computeEncoder.setBuffer(randomNumberBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(distributionBuffer, offset: 0, index: 2)
        
        if distributionType == .normal2 {
            computeEncoder.dispatchThreads([ (randomNumberSize + 1) / 2 ], threadsPerThreadgroup: 1)
        } else {
            computeEncoder.dispatchThreads([ randomNumberSize ], threadsPerThreadgroup: 1)
        }
        
        computeEncoder.endEncoding()
    }
    
    func custom_distributionMethod() -> Int {
        wrapGPUMethod(method: custom_distributionMethodRaw(commandBuffer:))
    }
}
