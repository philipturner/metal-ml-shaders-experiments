//
//  RandomApple.swift
//  DL4SExperiments
//
//  Created by Philip Turner on 11/24/21.
//

import MetalPerformanceShaders

extension RandomResources {
    
    fileprivate func makeMatrix(buffer: MTLBuffer, numElements: Int, offset: Int) -> MPSMatrix {
        let matrixDesc = MPSMatrixDescriptor(rows: 1,
                                             columns: numElements,
                                             rowBytes: numElements * MemoryLayout<UInt32>.stride,
                                             dataType: .float32)
        
        return MPSMatrix(buffer: buffer,
                         offset: offset * MemoryLayout<UInt32>.stride,
                         descriptor: matrixDesc)
    }
    
    func apple_randomMethodRaw(commandBuffer: MTLCommandBuffer) {
        let matrix = makeMatrix(buffer: randomNumberBuffer, numElements: randomNumberSize, offset: 0)
        
        let desc = MPSMatrixRandomDistributionDescriptor()
        desc.distributionType = .uniform
        desc.minimum = 0
        desc.maximum = 1
        
        let kernel = MPSMatrixRandomPhilox(device: device,
                                           destinationDataType: .float32,
                                           seed: 500,
                                           distributionDescriptor: desc)
        
        kernel.encode(commandBuffer: commandBuffer,
                      destinationMatrix: matrix)
    }
    
    func apple_randomMethod() -> Int {
        wrapGPUMethod(method: apple_randomMethodRaw(commandBuffer:))
    }
}
