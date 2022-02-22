//
//  ActivationResources.swift
//  DL4SExperiments
//
//  Created by Philip Turner on 11/24/21.
//

import ARHeadsetUtil
import Metal

class ActivationResources {
    var device: MTLDevice
    var commandQueue: MTLCommandQueue
    var activationPipelineState: MTLComputePipelineState
    
    init() {
        device = MTLCreateSystemDefaultDevice()!
        commandQueue = device.makeCommandQueue()!
        
        let library = try! device.makeDefaultLibrary(bundle: .safeModule)
        activationPipelineState = library.makeComputePipeline(ActivationResources.self, name: "testActivation")
    }
}
