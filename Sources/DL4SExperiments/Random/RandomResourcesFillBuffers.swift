//
//  RandomResourcesFillBuffers.swift
//  DL4SExperiments
//
//  Created by Philip Turner on 11/24/21.
//

import ARHeadsetUtil
import Metal

extension RandomResources {
    
    // Read
    
    func readValue8(index: Int, isDistribution: Bool = false) -> UInt8 {
        let pointer = (isDistribution ? distributionBuffer : randomNumberBuffer)
            .contents().assumingMemoryBound(to: UInt8.self)
        
        return pointer[index]
    }
    
    func readValue16(index: Int, isDistribution: Bool = false) -> UInt16 {
        let pointer = (isDistribution ? distributionBuffer : randomNumberBuffer)
            .contents().assumingMemoryBound(to: UInt16.self)
        
        return pointer[index]
    }
    
    func readValue32(index: Int, isDistribution: Bool = false) -> UInt32 {
        let pointer = (isDistribution ? distributionBuffer : randomNumberBuffer)
            .contents().assumingMemoryBound(to: UInt32.self)
        
        return pointer[index]
    }
    
    // Write
    
    func writeValue8(index: Int, value: UInt8, isDistribution: Bool = false) {
        let pointer = (isDistribution ? distributionBuffer : randomNumberBuffer)
            .contents().assumingMemoryBound(to: UInt8.self)
        
        pointer[index] = value
    }
    
    func writeValue16(index: Int, value: UInt16, isDistribution: Bool = false) {
        let pointer = (isDistribution ? distributionBuffer : randomNumberBuffer)
            .contents().assumingMemoryBound(to: UInt16.self)
        
        pointer[index] = value
    }
    
    func writeValue32(index: Int, value: UInt32, isDistribution: Bool = false) {
        let pointer = (isDistribution ? distributionBuffer : randomNumberBuffer)
            .contents().assumingMemoryBound(to: UInt32.self)
        
        pointer[index] = value
    }
    
    
    
}
