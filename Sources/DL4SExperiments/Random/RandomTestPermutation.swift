//
//  RandomTestPermutation.swift
//  DL4SExperiments
//
//  Created by Philip Turner on 11/24/21.
//

import ARHeadsetUtil

extension RandomResources {
    
    func testRandomExample(times: Int) throws {
        for _ in 0..<times {
            try testRandomExample()
        }
    }
    
    func testRandomExample() throws {
        var appleTimes: [Int] = []
        var customTimes: [Int] = []
        
        let numTests: Int = 50
        
        for _ in 0..<numTests {
            appleTimes.append(apple_randomMethod())
            customTimes.append(custom_randomMethod())
        }
        
        let appleAverage = appleTimes.reduce(0, +) / appleTimes.count
        let customAverage = customTimes.reduce(0, +) / customTimes.count
        
        print("Times (Apple, Custom)")
        print(appleAverage, customAverage)
    }
    
    
    // make a histogram test with min, max, # intervals
    // # data points already specified upon initialization
    
    // start with uniform distribution to debug graph presentation
    
    func testDistributionExample(min: Float, max: Float, intervals: Int) throws {
        _ = custom_randomMethod()
        _ = custom_distributionMethod()
        
        let randomVals = (0..<randomNumberSize).map { i -> Float in
            if generatingBF16 {
                let intVal = readValue16(index: i, isDistribution: true)
                return Float(unsafeBitCast(intVal, to: BFloat16.self))
            } else {
                let intVal = readValue32(index: i, isDistribution: true)
                return Float(bitPattern: intVal)
            }
        }
        
//        print(randomVals)
        
        let outsideNum = randomVals.reduce(into: 0) { if $1 < min || $1 > max { $0 += 1 } }
        let filteredVals = randomVals.filter { $0 >= min && $0 <= max }.map { ($0 - min) / (max - min) }
        assert(filteredVals.count + outsideNum == randomVals.count)
        
//        print(filteredVals)
        
//        print(outsideNum)
        let histCodes = filteredVals.map{ $0 * Float(intervals) }.map { Swift.max(0, Swift.min(Int($0), intervals - 1)) }
//        print(histCodes)
        
        // need to show each interval's range
        
        func normalizeStrings(_ strings: inout [String]) {
            let maxLength = strings.reduce(into: 0) { $0 = Swift.max($0, $1.count) }
            
            for i in 0..<strings.count {
                strings[i] += String(repeating: " ", count: maxLength - strings[i].count)
            }
        }
        
        var intervalDistributions = (0..<intervals).map { i -> String in
            "Segment \(i)"
        }
        
        normalizeStrings(&intervalDistributions)
        
        intervalDistributions = intervalDistributions.map{ $0 + ": " }
        let intervalSize = (max - min) / Float(intervals)
        
        for i in 0..<intervals {
            let minVal = min + intervalSize * Float(i)
            let maxVal = minVal + intervalSize
            
            let strVal1 = String(format: "%.2f", minVal)
            let strVal2 = String(format: "%.2f", maxVal)
            
            intervalDistributions[i] += "\(strVal1) - \(strVal2)"
        }
        
        intervalDistributions.append("Outside Bounds:")
        normalizeStrings(&intervalDistributions)
        
        let outsideStr = intervalDistributions.removeLast()
        var counts = [Int](repeating: 0, count: intervals)
        
        for code in histCodes {
            counts[code] += 1
        }
        
        for i in 0..<intervals {
            let count = counts[i]
            let proportion = Float(count) / Float(randomVals.count)
            let percent = proportion * 100
                                                  
            intervalDistributions[i] += "\t" + String(format: "%.2f", percent) + "%"
        }
        
        print("Histogram")
        for segment in intervalDistributions {
            print(segment)
        }
        
        let outsidePercent = Float(outsideNum) / Float(randomVals.count) * 100
        let outsidePercentStr = "\t" + String(format: "%.2f", outsidePercent) + "%"
        
        print(outsideStr + outsidePercentStr)
    }
}
