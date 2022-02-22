//
//  MMXApple.swift
//  DL4SExperiments
//
//  Created by Philip Turner on 11/24/21.
//

import MetalPerformanceShaders
import Accelerate

extension MMXResources {
    
    fileprivate func makeMatrix(buffer: MTLBuffer) -> MPSMatrix {
        let numRows = isTranspose ? self.numCols : self.numRows
        let numCols = isTranspose ? self.numRows : self.numCols
        
        let matrixDesc = MPSMatrixDescriptor(rows: numRows,
                                             columns: numCols,
                                             rowBytes: (~3 & (numCols + 3)) * MemoryLayout<Float>.stride,
                                             dataType: .float32)
        
        return MPSMatrix(buffer: buffer, descriptor: matrixDesc)
    }
    
    fileprivate func makeVector(buffer: MTLBuffer, size: Int) -> MPSVector {
        let vectorDesc = MPSVectorDescriptor(length: size,
                                             vectors: 1,
                                             vectorBytes: (~3 & (size + 3)) * MemoryLayout<Float>.stride,
                                             dataType: .float32)
        
        return MPSVector(buffer: buffer, descriptor: vectorDesc)
    }
    
    func makeSmallMatrixTranspose(buffer: MTLBuffer, size: Int) -> MPSMatrix {
        let matrixDesc = MPSMatrixDescriptor(rows: numInstances,
                                             columns: size,
                                             rowBytes: (~3 & (size + 3)) * MemoryLayout<Float>.stride,
                                             dataType: .float32)
        
        return MPSMatrix(buffer: buffer, descriptor: matrixDesc)
    }
    
    func makeSmallMatrix(buffer: MTLBuffer, size: Int) -> MPSMatrix {
        let matrixDesc = MPSMatrixDescriptor(rows: size,
                                             columns: numInstances,
                                             rowBytes: numInstances * MemoryLayout<Float>.stride,
                                             dataType: .float32)
        
        return MPSMatrix(buffer: buffer, descriptor: matrixDesc)
    }
    
    
    
    func gpu_gemvMethodRaw(commandBuffer: MTLCommandBuffer) {
        let matrix1 = makeMatrix(buffer: buffer1)
        let vector1 = makeVector(buffer: buffer2, size: numCols)
        let vector2 = makeVector(buffer: buffer3, size: numRows)
        
        let kernel = MPSMatrixVectorMultiplication(device: device,
                                                   transpose: isTranspose,
                                                   rows: numRows,
                                                   columns: numCols,
                                                   alpha: 1, beta: 0)
        
        kernel.encode(commandBuffer: commandBuffer,
                      inputMatrix: matrix1,
                      inputVector: vector1,
                      resultVector: vector2)
    }
    
    func gpu_gemvMethod() -> Int {
        wrapGPUMethod(method: gpu_gemvMethodRaw)
    }
    
    func gpu_gemmMethodRaw(commandBuffer: MTLCommandBuffer) {
        let matrix1 = makeMatrix(buffer: buffer1)
        
        let matrix2 = makeSmallMatrixTranspose(buffer: buffer2, size: numCols)
        let matrix3 = makeSmallMatrix(buffer: buffer3, size: numRows)
        
        let kernel = MPSMatrixMultiplication(device: device,
                                             transposeLeft: isTranspose,
                                             transposeRight: true,
                                             resultRows: numRows,
                                             resultColumns: numInstances,
                                             interiorColumns: numCols,
                                             alpha: 1, beta: 0)
        
        kernel.encode(commandBuffer: commandBuffer,
                      leftMatrix: matrix1,
                      rightMatrix: matrix2,
                      resultMatrix: matrix3)
    }
    
    func gpu_gemmMethod() -> Int {
        wrapGPUMethod(method: gpu_gemmMethodRaw)
    }
    
}
