//
//  Random.swift
//  DL4SExperiments
//
//  Created by Philip Turner on 11/24/21.
//

import ARHeadsetUtil

struct PhiloxRandomNumberGenerator {
    private var counter: UInt = 0
    private var key: simd_uint2
    
    // generates 2 64-bit values or 4 32-bit values at a time???
    // read research paper again to understand
    
//    private var useNextValue = false
//    private var nextValue: UInt = 0
    
    private func bump(key: simd_uint2) -> simd_uint2 {
        let bumpConstantHi: UInt32 = 0x9E37_79B9
        let bumpConstantLo: UInt32 = 0xBB67_AE85
        return [key[0] &+ bumpConstantLo, key[1] &+ bumpConstantHi]
    }
    
    private func round(ctr: simd_uint4, key: simd_uint2) -> simd_uint4 {
        let roundConstant0: UInt = 0xD251_1F53
        let roundConstant1: UInt = 0xCD9E_8D57
        
        let product0 = roundConstant0 &* UInt(ctr[3])
        let hi0 = UInt32(truncatingIfNeeded: product0 >> 32)
        let lo0 = UInt32(truncatingIfNeeded: (product0 & 0x0000_0000_FFFF_FFFF))
        
        let product1 = roundConstant1 &* UInt(ctr[1])
        let hi1 = UInt32(truncatingIfNeeded: product1 >> 32)
        let lo1 = UInt32(truncatingIfNeeded: (product1 & 0x0000_0000_FFFF_FFFF))
        
        return [lo0, hi0 ^ ctr[0] ^ key[0], lo1, hi1 ^ ctr[2] ^ key[1]]
    }
    
    private func random(forCtr initialCtr: simd_uint4, key initialKey: simd_uint2) -> simd_uint4 {
        var ctr = initialCtr
        var key = initialKey
        // 10 rounds
        // R1
        ctr = round(ctr: ctr, key: key)
        // R2
        key = bump(key: key)
        ctr = round(ctr: ctr, key: key)
        // R3
        key = bump(key: key)
        ctr = round(ctr: ctr, key: key)
        // R4
        key = bump(key: key)
        ctr = round(ctr: ctr, key: key)
        // R5
        key = bump(key: key)
        ctr = round(ctr: ctr, key: key)
        // R6
        key = bump(key: key)
        ctr = round(ctr: ctr, key: key)
        // R7
        key = bump(key: key)
        ctr = round(ctr: ctr, key: key)
        // R8
        key = bump(key: key)
        ctr = round(ctr: ctr, key: key)
        // R9
        key = bump(key: key)
        ctr = round(ctr: ctr, key: key)
        // R10
        key = bump(key: key)
        ctr = round(ctr: ctr, key: key)
        
        return ctr
    }
    
    public init(uint64Seed: UInt) {
        key = unsafeBitCast(uint64Seed, to: simd_uint2.self)
    }
    
    public mutating func next() -> simd_uint4 {
        let counterVector = unsafeBitCast(simd_ulong2(counter, 0), to: simd_uint4.self)
        let output = random(forCtr: counterVector, key: key)
        counter += 1
        
        return output
    }
}

