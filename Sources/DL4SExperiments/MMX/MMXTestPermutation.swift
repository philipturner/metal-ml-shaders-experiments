//
//  MMXTestPermutation.swift
//  DL4SExperiments
//
//  Created by Philip Turner on 11/24/21.
//

import ARHeadsetUtil

extension MMXResources {
    
    func wrapTestMethod(customMethod: () -> Int, appleMethod: () -> Int) throws {
        let loopSize = 50
        
        var appleTimes = [Int]()
        var customTimes = [Int]()
        
        fillBuffer(id: 3, with: 15)
        
        for i in 0..<loopSize {
            if i == loopSize - 1 {
                fillBuffer(id: 0, with: 2)
                fillBuffer(id: 1, with: 3)
                fillBuffer(id: 2, with: -1)
            }
            
            let custom = customMethod()
            let customVal = (i == loopSize - 1) ? readValue(id: 2, fromApple: false) : nil
            
            if i == loopSize - 1 {
                fillBuffer(id: 0, with: 2)
                fillBuffer(id: 1, with: 3)
                fillBuffer(id: 2, with: -1)
            }
            
            let apple = appleMethod()
            let appleVal = (i == loopSize - 1) ? readValue(id: 2, fromApple: true) : nil
            
            if i == loopSize - 1 {
                print("\(apple)\t\(custom)")
                print("\(appleVal!)\t\(customVal!)")
            }
            
            appleTimes.append(apple)
            customTimes.append(custom)
        }
        
        print("apple:  \(appleTimes .reduce(0, +) / loopSize / 10)")
        print("custom: \(customTimes.reduce(0, +) / loopSize / 10)")
        print("time saved by custom: \(Float(appleTimes .reduce(0, +) / loopSize - customTimes.reduce(0, +) / loopSize) / 10)")
    }
    
    func testSGEMVExample() throws {
        try wrapTestMethod(customMethod: custom_sgemvMethod, appleMethod: gpu_gemmMethod)
    }
    
    func testSGEMMExample() throws {
        try wrapTestMethod(customMethod: custom_sgemmMethod, appleMethod: gpu_gemmMethod)
    }
    
}
