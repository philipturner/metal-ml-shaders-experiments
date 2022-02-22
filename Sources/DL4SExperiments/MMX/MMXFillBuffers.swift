//
//  MMXFillBuffers.swift
//  DL4SExperiments
//
//  Created by Philip Turner on 11/24/21.
//

import ARHeadsetUtil
import Metal

extension MMXResources {
    
    func zeroOutBuffers() {
        for i in 0..<3 {
            fillBuffer(id: i, with: 0)
        }
    }
    
    func readBuffer(id: Int) -> UnsafeMutablePointer<Float> {
        let buffer = [buffer1, buffer2, buffer3][id]
        return buffer.contents().assumingMemoryBound(to: Float.self)
    }
    
    func getReadLocation(fromApple: Bool) -> Int {
        fromApple ? readLocationApple : readLocationCustom
    }
    
    func readValue(id: Int, fromApple: Bool, index: Int? = nil) -> Float {
        let pointer = [buffer1, buffer2, buffer3][id].contents()
        let readLocation = getReadLocation(fromApple: fromApple)
        
        if isBF16(id: id) {
            return Float(pointer.assumingMemoryBound(to: BFloat16.self)[index ?? readLocation])
        } else {
            return pointer.assumingMemoryBound(to: Float.self)[index ?? readLocation]
        }
    }
    
    func writeValue(id: Int, fromApple: Bool, index: Int? = nil, value: Float) {
        let pointer = [buffer1, buffer2, buffer3][id].contents()
        let readLocation = getReadLocation(fromApple: fromApple)
        
        if isBF16(id: id) {
            print("did get bfloat")
            pointer.assumingMemoryBound(to: BFloat16.self)[index ?? readLocation] = .init(value)
        } else {
            pointer.assumingMemoryBound(to: Float.self)[index ?? readLocation] = value
        }
    }
    
    func fillMaskBuffer(value: UInt8, bytesPerRow: Int, numRows: Int) {
        let numBytes = bytesPerRow * numRows
        let pointer = buffer4.contents()
        memset(pointer, Int32(value), numBytes)
    }
    
    func isBF16(id: Int) -> Bool {
        switch id {
        case 0:
            return matrixIsBF16
        case 1:
            return inVectorIsBF16
        case 2:
            return outVectorIsBF16
        default:
            fatalError()
        }
    }
    
    func fillBuffer(id: Int, with value: Float) {
        guard id < 3 else {
            let rowElementStride = isTranspose ? numRows : numCols
            let rowDimension     = isTranspose ? numCols : numRows
            fillMaskBuffer(value: UInt8(value), bytesPerRow: (7 + rowElementStride) / 8, numRows: rowDimension)
            
            return
        }
        
        let buffer = [buffer1, buffer2, buffer3][id]
        let pointer = readBuffer(id: id)
        let length = buffer.length / MemoryLayout<Float>.stride
        
        if !isBF16(id: id) {
            memset_pattern4(pointer, value, count: length)
        } else {
            let pointer2 = UnsafeMutablePointer<simd_bfloat16_2>(OpaquePointer(pointer))
                                                                 
            let bf16_value = simd_bfloat16_2([ value, value ])
            memset_pattern4(pointer2, bf16_value, count: ~1 & (length + 1))
        }
    }
    
}
