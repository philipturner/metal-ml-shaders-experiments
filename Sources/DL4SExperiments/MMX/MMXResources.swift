//
//  MMXResources.swift
//  DL4SExperiments
//
//  Created by Philip Turner on 11/24/21.
//

import ARHeadsetUtil
import Metal

class MMXResources {
    var device: MTLDevice
    var commandQueue: MTLCommandQueue
    
    var numRows: Int = 1024 // 492
    var numCols: Int = 1024 // 372
    var numInstances: Int = 4 // 90
    var textureSize: Int
    
    var readLocationCustom: Int
    var readLocationApple: Int
    
    // function constants (begin)
    
    var isTranspose: Bool
    var isSparse: Bool = false
    
    var matrixIsBF16: Bool
    var inVectorIsBF16: Bool = false
    var outVectorIsBF16: Bool = false
    
    var usingNonuniformDispatch: Bool
    var supportsSIMDPermute: Bool
    var tileSize: Int
    var isAligned: Bool
    
    // function constants (end)
    
    var buffer1: MTLBuffer
    var buffer2: MTLBuffer
    var buffer3: MTLBuffer
    var buffer4: MTLBuffer
    
    var sgemvPipelineState: MTLComputePipelineState
    var sgemmPipelineState: MTLComputePipelineState
    // for the large sgemm pipeline state, mark that threadgroup size is multiple of execution width
    
    var icb: MTLIndirectCommandBuffer
    
    init(isTranspose: Bool, matrixIsBF16: Bool, usingNonuniformDispatch: Bool, supportsSIMDPermute: Bool) {
        device = MTLCreateSystemDefaultDevice()!
        commandQueue = device.makeCommandQueue()!
        
        textureSize = roundUpToPowerOf2(max(max(numRows, numCols), numInstances))
        readLocationCustom = (~3 & (numRows + 3)) * (numInstances - 1)
        readLocationCustom += numRows - 1
        readLocationApple = numRows * numInstances - 1
        
        self.isTranspose = isTranspose
        self.matrixIsBF16 = matrixIsBF16
        self.usingNonuniformDispatch = usingNonuniformDispatch
        self.supportsSIMDPermute = supportsSIMDPermute
        
        let desc = MTLIndirectCommandBufferDescriptor()
        if #available(macOS 11.0, *) {
            desc.commandTypes = .concurrentDispatch
        } else {
            // Fallback on earlier versions
        }
        if #available(macOS 11.0, *) {
            desc.maxKernelBufferBindCount = 1
        } else {
            // Fallback on earlier versions
        }
        
        icb = device.makeIndirectCommandBuffer(descriptor: desc, maxCommandCount: 1)!
        
        
        
        let library = try! device.makeDefaultLibrary(bundle: .safeModule)
        
        let constants = MTLFunctionConstantValues()
        constants.setConstantValue(&self.isTranspose,             type: .bool, index: 0)
        constants.setConstantValue(&isSparse,                     type: .bool, index: 1)
        
        constants.setConstantValue(&self.matrixIsBF16,            type: .bool, index: 2)
        constants.setConstantValue(&inVectorIsBF16,               type: .bool, index: 3)
        constants.setConstantValue(&outVectorIsBF16,              type: .bool, index: 4)
        
        constants.setConstantValue(&self.usingNonuniformDispatch, type: .bool, index: 5)
        constants.setConstantValue(&self.supportsSIMDPermute,     type: .bool, index: 6)
        
        let function = try! library.makeFunction(name: "sgemv", constantValues: constants)
        sgemvPipelineState = library.makeComputePipeline(MMXResources.self, name: "sgemv", function: function)
        
        
        
        let descriptor = MTLComputePipelineDescriptor()
        descriptor.optLabel = "MMXResources sgemm_small Pipeline"
        
        let executionWidth = sgemvPipelineState.threadExecutionWidth
        let deviceCopy = self.device
        
        func makeSGEMMPipeline(tileSize: Int, isAligned: Bool) -> MTLComputePipelineState {
            var tileSizeCopy = tileSize
            var isAlignedCopy = isAligned
            constants.setConstantValue(&tileSizeCopy,  type: .ushort, index: 7)
            constants.setConstantValue(&isAlignedCopy, type: .bool,   index: 8)
            
            let function = try! library.makeFunction(name: "sgemm_small", constantValues: constants)
            descriptor.computeFunction = function
            descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = tileSize * tileSize >= executionWidth
            
            return deviceCopy.makeComputePipelineState(descriptor: descriptor)
        }
        
        let longDimension = numRows
        let shortDimensions = simd_long2(numCols, numInstances)
        
        func getAligned(tileSize: Int) -> Bool {
            (longDimension & (tileSize * 4 - 1) == 0) &&
            all(shortDimensions & (tileSize - 1) .== 0)
        }
        
        tileSize = min(min(numRows, numCols), numInstances)
        tileSize = max(4, min(roundUpToPowerOf2(tileSize), 32))
        isAligned = getAligned(tileSize: tileSize)
        
        sgemmPipelineState = makeSGEMMPipeline(tileSize: tileSize, isAligned: isAligned)
        
        while sgemmPipelineState.threadExecutionWidth < tileSize ||
              sgemmPipelineState.maxTotalThreadsPerThreadgroup < tileSize * tileSize
        {
            tileSize /= 2
            isAligned = getAligned(tileSize: tileSize)
            
            sgemmPipelineState = makeSGEMMPipeline(tileSize: tileSize, isAligned: isAligned)
        }
        
        
        
        let bufferSize = textureSize * textureSize * 4
        
        buffer1 = device.makeBuffer(length: bufferSize, options: .storageModeShared)!
        buffer2 = device.makeBuffer(length: bufferSize, options: .storageModeShared)!
        buffer3 = device.makeBuffer(length: bufferSize, options: .storageModeShared)!
        buffer4 = device.makeBuffer(length: bufferSize, options: .storageModeShared)!
    }
    
    func wrapGPUMethod(method: (MTLCommandBuffer) -> Void) -> Int {
        let commandBuffer = commandQueue.makeDebugCommandBuffer()
        
        method(commandBuffer)
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        return Int((commandBuffer.gpuEndTime - commandBuffer.gpuStartTime) * 1e7)
    }
}
