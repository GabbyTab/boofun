"""
Cryptographic Analysis Demo for BooFun Library

This example demonstrates cryptographic analysis tools for Boolean functions,
useful for S-box design and block cipher analysis.

Features covered:
- Nonlinearity and bent function detection
- Walsh transform and spectrum analysis
- Linear Approximation Table (LAT)
- Difference Distribution Table (DDT)
- Algebraic immunity
- S-box comprehensive analysis
"""

import numpy as np
import boofun as bf
from boofun.analysis.cryptographic import (
    nonlinearity,
    is_bent,
    is_balanced,
    walsh_transform,
    walsh_spectrum,
    algebraic_degree,
    algebraic_normal_form,
    correlation_immunity,
    resiliency,
    strict_avalanche_criterion,
    linear_approximation_table,
    difference_distribution_table,
    SBoxAnalyzer,
)


def demo_basic_cryptographic_properties():
    """Demonstrate basic cryptographic measures."""
    print("=" * 60)
    print("Basic Cryptographic Properties")
    print("=" * 60)
    
    # Create some test functions
    xor = bf.parity(4)  # XOR is linear
    majority = bf.majority(5)  # Majority is nonlinear
    bent_tt = [0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0]  # A bent function
    bent_f = bf.create(bent_tt)
    
    print("\n1. Nonlinearity (distance to nearest affine function)")
    print("-" * 50)
    print(f"   Parity (linear):   nonlinearity = {nonlinearity(xor)}")
    print(f"   Majority:          nonlinearity = {nonlinearity(majority)}")
    print(f"   Bent function:     nonlinearity = {nonlinearity(bent_f)}")
    
    print("\n2. Bent Function Detection")
    print("-" * 50)
    print(f"   Parity is bent:    {is_bent(xor)}")
    print(f"   Majority is bent:  {is_bent(majority)}")
    print(f"   Bent func is bent: {is_bent(bent_f)}")
    
    print("\n3. Balance Check")
    print("-" * 50)
    print(f"   Parity balanced:   {is_balanced(xor)}")
    print(f"   AND balanced:      {is_balanced(bf.AND(4))}")
    

def demo_walsh_analysis():
    """Demonstrate Walsh transform analysis."""
    print("\n" + "=" * 60)
    print("Walsh Transform Analysis")
    print("=" * 60)
    
    f = bf.majority(3)
    
    print("\n1. Walsh Transform (all coefficients)")
    print("-" * 50)
    wt = walsh_transform(f)
    print(f"   Walsh coefficients: {wt}")
    
    print("\n2. Walsh Spectrum (unique absolute values)")
    print("-" * 50)
    spectrum = walsh_spectrum(f)
    print(f"   Spectrum: {spectrum}")
    
    print("\n3. Algebraic Degree")
    print("-" * 50)
    print(f"   Majority(3) degree: {algebraic_degree(f)}")
    print(f"   Parity(4) degree:   {algebraic_degree(bf.parity(4))}")
    print(f"   AND(3) degree:      {algebraic_degree(bf.AND(3))}")


def demo_advanced_cryptographic():
    """Demonstrate advanced cryptographic measures."""
    print("\n" + "=" * 60)
    print("Advanced Cryptographic Measures")
    print("=" * 60)
    
    f = bf.majority(5)
    
    print("\n1. Correlation Immunity")
    print("-" * 50)
    ci = correlation_immunity(f)
    print(f"   Majority(5) correlation immunity: {ci}")
    
    print("\n2. Resiliency")
    print("-" * 50)
    res = resiliency(f)
    print(f"   Majority(5) resiliency: {res}")
    
    print("\n3. Strict Avalanche Criterion (SAC)")
    print("-" * 50)
    sac = strict_avalanche_criterion(f)
    print(f"   Majority(5) SAC: {sac}")


def demo_sbox_analysis():
    """Demonstrate S-box analysis."""
    print("\n" + "=" * 60)
    print("S-Box Analysis")
    print("=" * 60)
    
    # A simple 4-bit S-box (similar to a reduced AES S-box)
    sbox = [0xE, 0x4, 0xD, 0x1, 0x2, 0xF, 0xB, 0x8,
            0x3, 0xA, 0x6, 0xC, 0x5, 0x9, 0x0, 0x7]
    
    print(f"\nS-box: {[hex(x) for x in sbox]}")
    
    print("\n1. Linear Approximation Table (LAT)")
    print("-" * 50)
    lat = linear_approximation_table(sbox)
    print(f"   LAT shape: {lat.shape}")
    print(f"   Max bias (excluding 0,0): {np.max(np.abs(lat[1:, 1:]))}")
    
    print("\n2. Difference Distribution Table (DDT)")
    print("-" * 50)
    ddt = difference_distribution_table(sbox)
    print(f"   DDT shape: {ddt.shape}")
    print(f"   Max differential (excluding 0): {np.max(ddt[1:, :])}")
    
    print("\n3. Comprehensive S-box Analysis")
    print("-" * 50)
    analyzer = SBoxAnalyzer(sbox)
    summary = analyzer.summary()
    print(f"   Bits: {summary['bits']}")
    print(f"   Is bijective: {summary['is_bijective']}")
    print(f"   Nonlinearity: {summary['nonlinearity']}")
    print(f"   Differential uniformity: {summary['differential_uniformity']}")
    print(f"   Linearity: {summary['linearity']}")


def demo_algebraic_properties():
    """Demonstrate algebraic properties analysis."""
    print("\n" + "=" * 60)
    print("Algebraic Properties")
    print("=" * 60)
    
    f = bf.majority(3)
    
    print("\n1. Algebraic Normal Form (ANF)")
    print("-" * 50)
    anf = algebraic_normal_form(f)
    print(f"   Majority(3) ANF terms: {len(anf)} terms")
    for term, coeff in sorted(anf.items(), key=lambda x: len(x[0])):
        if coeff:
            vars_str = " AND ".join([f"x{i}" for i in term]) if term else "1"
            print(f"      {vars_str}")


def main():
    """Run all cryptographic analysis demos."""
    print("\n" + "=" * 60)
    print("BooFun Cryptographic Analysis Demo")
    print("=" * 60)
    
    demo_basic_cryptographic_properties()
    demo_walsh_analysis()
    demo_advanced_cryptographic()
    demo_sbox_analysis()
    demo_algebraic_properties()
    
    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
