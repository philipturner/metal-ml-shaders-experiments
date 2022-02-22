//
//  S4TF_Random.swift
//  DL4SExperiments
//
//  Created by Philip Turner on 11/24/21.
//

import Foundation

private typealias UInt32x2 = (UInt32, UInt32)
private typealias UInt32x4 = (UInt32, UInt32, UInt32, UInt32)

public struct S4TF_PhiloxRandomNumberGenerator {
  public static var global = S4TF_PhiloxRandomNumberGenerator(uint64Seed: UInt64(time(nil)))

  private var ctr: UInt64 = 0
  private let key: UInt32x2

  // Since we generate two 64-bit values at a time, we only need to run the
  // generator every other invocation.
  private var useNextValue = false
  private var nextValue: UInt64 = 0

  private func bump(key: UInt32x2) -> UInt32x2 {
    let bumpConstantHi: UInt32 = 0x9E37_79B9
    let bumpConstantLo: UInt32 = 0xBB67_AE85
    return (key.0 &+ bumpConstantHi, key.1 &+ bumpConstantLo)
  }

  private func round(ctr: UInt32x4, key: UInt32x2) -> UInt32x4 {
    let roundConstant0: UInt64 = 0xD251_1F53
    let roundConstant1: UInt64 = 0xCD9E_8D57

    let product0: UInt64 = roundConstant0 &* UInt64(ctr.0)
    let hi0 = UInt32(truncatingIfNeeded: product0 >> 32)
    let lo0 = UInt32(truncatingIfNeeded: (product0 & 0x0000_0000_FFFF_FFFF))

    let product1: UInt64 = roundConstant1 &* UInt64(ctr.2)
    let hi1 = UInt32(truncatingIfNeeded: product1 >> 32)
    let lo1 = UInt32(truncatingIfNeeded: (product1 & 0x0000_0000_FFFF_FFFF))

    return (hi1 ^ ctr.1 ^ key.0, lo1, hi0 ^ ctr.3 ^ key.1, lo0)
  }

  private func random(forCtr initialCtr: UInt32x4, key initialKey: UInt32x2) -> UInt32x4 {
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

  public init(uint64Seed seed: UInt64) {
    key = seed.vector2
  }

  public init(seed: [UInt8]) {
    precondition(seed.count > 0, "Length of seed must be positive")
    precondition(seed.count <= 8, "Length of seed must be at most 8")
    var combinedSeed: UInt64 = 0
    for (i, byte) in seed.enumerated() {
      combinedSeed += UInt64(byte) << UInt64(8 * i)
    }
    self.init(uint64Seed: combinedSeed)
  }

  public mutating func next() -> UInt64 {
    if useNextValue {
      useNextValue = false
      return nextValue
    }
    let (this, next) = makeUInt64Pair(random(forCtr: ctr.vector4, key: key))
    useNextValue = true
    nextValue = next
    ctr += 1
    
    return this
  }
}

/// Private helpers.
extension UInt64 {
  fileprivate var vector2: UInt32x2 {
    let msb = UInt32(truncatingIfNeeded: self >> 32)
    let lsb = UInt32(truncatingIfNeeded: self & 0x0000_0000_FFFF_FFFF)
    return (msb, lsb)
  }

  fileprivate var vector4: UInt32x4 {
    let msb = UInt32(truncatingIfNeeded: self >> 32)
    let lsb = UInt32(truncatingIfNeeded: self & 0x0000_0000_FFFF_FFFF)
    return (0, 0, msb, lsb)
  }

  fileprivate init(vector: UInt32x2) {
    self = (UInt64(vector.0) << 32) + UInt64(vector.1)
  }
}

private func makeUInt64Pair(_ vector: UInt32x4) -> (UInt64, UInt64) {
  let a = (UInt64(vector.0) << 32) + UInt64(vector.1)
  let b = (UInt64(vector.2) << 32) + UInt64(vector.3)
  return (a, b)
}
