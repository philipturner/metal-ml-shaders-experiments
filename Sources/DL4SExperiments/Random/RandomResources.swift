//
//  RandomResources.swift
//  DL4SExperiments
//
//  Created by Philip Turner on 11/24/21.
//

import ARHeadsetUtil
import Metal

class RandomResources {
    var device: MTLDevice
    var commandQueue: MTLCommandQueue
    
    var randomNumberSize: Int
    var randomNumberBuffer: MTLBuffer
    var distributionBuffer: MTLBuffer
    var key: UInt64 = 500
    
    enum DistributionType: UInt16 {
        case uniform = 0
        case normal2 = 1
        case bernoulli = 2
    }
    
    var supportsLong: Bool
    var generatingBF16: Bool
    var distributionType: DistributionType
    
    var paramA: Float!
    var paramB: Float!
    
    var tapePipelineState: MTLComputePipelineState
    var distributionPipelineState: MTLComputePipelineState
    
    init(randomNumberSize: Int, supportsLong: Bool, generatingBF16: Bool, distributionType: DistributionType, paramA: Float? = nil, paramB: Float? = nil) {
        self.randomNumberSize = randomNumberSize
        self.supportsLong = supportsLong
        self.generatingBF16 = generatingBF16
        self.distributionType = distributionType
        self.paramA = paramA
        self.paramB = paramB
        
        device = MTLCreateSystemDefaultDevice()!
        commandQueue = device.makeCommandQueue()!
        
        let bufferSize = randomNumberSize * MemoryLayout<UInt32>.stride
        randomNumberBuffer = device.makeBuffer(length: bufferSize)!
        distributionBuffer = device.makeBuffer(length: bufferSize)!
        
        let library = try! device.makeDefaultLibrary(bundle: .safeModule)
        
        
        
        let constants = MTLFunctionConstantValues()
        constants.setConstantValue(&self.supportsLong, type: .bool, index: 0)
        
        var function = try! library.makeFunction(name: "generateRandomTape", constantValues: constants)
        tapePipelineState = library.makeComputePipeline(RandomResources.self, name: "generateRandomTape", function: function)
        
        var distCode = distributionType.rawValue
        constants.setConstantValue(&self.generatingBF16, type: .bool,   index: 1)
        constants.setConstantValue(&distCode,            type: .ushort, index: 2)
        
        function = try! library.makeFunction(name: "generateRandomDistribution", constantValues: constants)
        distributionPipelineState = library.makeComputePipeline(RandomResources.self, name: "generateRandomDistribution", function: function)
    }
    
    func wrapGPUMethod(method: (MTLCommandBuffer) -> Void) -> Int {
        let commandBuffer = commandQueue.makeDebugCommandBuffer()
        
        method(commandBuffer)
        
        commandBuffer.commit()
        commandBuffer.waitUntilCompleted()
        
        return Int((commandBuffer.gpuEndTime - commandBuffer.gpuStartTime) * 1e7)
    }
}
