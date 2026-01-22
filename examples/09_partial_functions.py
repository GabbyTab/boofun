"""
Partial Boolean Functions Demo for BooFun Library

This example demonstrates working with partial Boolean functions,
useful for streaming data, incomplete specifications, and large functions
where you can't store the full truth table.

Features covered:
- Creating partial functions
- Streaming/incremental specification
- Hex string I/O (thomasarmel-compatible)
- Storage hints for large functions
- Confidence-based evaluation
"""

import numpy as np

import boofun as bf


def demo_partial_creation():
    """Demonstrate creating partial Boolean functions."""
    print("=" * 60)
    print("Creating Partial Boolean Functions")
    print("=" * 60)

    print("\n1. Create with known values")
    print("-" * 50)
    # Create a partial function with some known outputs
    p = bf.partial(n=4, known_values={0: False, 1: True, 15: True})
    print(f"   Created partial function with n=4")
    print(f"   Known values: {p.num_known} / {2**4}")
    print(f"   Completeness: {p.completeness:.2%}")

    print("\n2. Query known vs unknown")
    print("-" * 50)
    print(f"   p[0] (known):   {p[0]}")
    print(f"   p[1] (known):   {p[1]}")
    print(f"   p[5] (unknown): {p[5]}")  # Returns None


def demo_streaming():
    """Demonstrate streaming/incremental specification."""
    print("\n" + "=" * 60)
    print("Streaming Boolean Function Specification")
    print("=" * 60)

    print("\n1. Incremental addition")
    print("-" * 50)
    p = bf.partial(n=5)
    print(f"   Initial: {p.num_known} known values")

    # Add values one at a time (streaming)
    p.add(0, False)
    p.add(1, True)
    p.add(2, True)
    print(f"   After 3 adds: {p.num_known} known values")

    print("\n2. Batch addition")
    print("-" * 50)
    p.add_batch({10: True, 11: False, 12: True, 13: False, 14: True})
    print(f"   After batch: {p.num_known} known values")
    print(f"   Completeness: {p.completeness:.2%}")

    print("\n3. Convert to full function when ready")
    print("-" * 50)
    # Fill unknown values with False
    f = p.to_function(fill_unknown=False)
    print(f"   Converted to BooleanFunction with {f.n_vars} variables")


def demo_hex_io():
    """Demonstrate hex string I/O."""
    print("\n" + "=" * 60)
    print("Hex String I/O (thomasarmel-compatible)")
    print("=" * 60)

    print("\n1. Create from hex string")
    print("-" * 50)
    # 4-bit bent function from thomasarmel examples
    f = bf.from_hex("ac90", n=4)
    print(f"   Created from 'ac90': n={f.n_vars}")
    print(f"   Truth table: {list(f.get_representation('truth_table'))}")

    print("\n2. Larger function from hex")
    print("-" * 50)
    # 6-bit function
    f6 = bf.from_hex("0113077C165E76A8", n=6)
    print(f"   Created from 64-bit hex: n={f6.n_vars}")

    print("\n3. Export to hex")
    print("-" * 50)
    maj = bf.majority(3)
    hex_str = bf.to_hex(maj)
    print(f"   Majority(3) as hex: {hex_str}")

    # Round-trip verification
    maj2 = bf.from_hex(hex_str, n=3)
    print(
        f"   Round-trip matches: {list(maj.get_representation('truth_table')) == list(maj2.get_representation('truth_table'))}"
    )


def demo_storage_hints():
    """Demonstrate storage hints for different use cases."""
    print("\n" + "=" * 60)
    print("Storage Hints for Large Functions")
    print("=" * 60)

    print("\n1. Dense storage (default)")
    print("-" * 50)
    tt = [0, 1, 1, 0, 1, 0, 0, 1]
    f_dense = bf.create(tt, storage="dense")
    print(f"   Dense: standard array storage")

    print("\n2. Packed storage (1 bit per entry)")
    print("-" * 50)
    f_packed = bf.create(tt, storage="packed")
    print(f"   Packed: 8x memory reduction for large n")

    print("\n3. Sparse storage (for mostly-0 or mostly-1)")
    print("-" * 50)
    # Create a sparse function (mostly False, few True)
    sparse_tt = [0] * 256
    sparse_tt[42] = 1
    sparse_tt[137] = 1
    f_sparse = bf.create(sparse_tt, storage="sparse")
    print(f"   Sparse: only stores exceptions")

    print("\n4. Auto storage (selects best)")
    print("-" * 50)
    f_auto = bf.create(tt, storage="auto")
    print(f"   Auto: picks based on size and sparsity")

    print("\n5. Lazy/oracle storage (compute on demand)")
    print("-" * 50)

    def my_oracle(x):
        """Example: parity function as oracle."""
        if isinstance(x, int):
            return bin(x).count("1") % 2
        return sum(x) % 2

    f_lazy = bf.create(my_oracle, n=20, storage="lazy")
    print(f"   Lazy: n=20 would need 1M entries, but oracle computes on demand")
    print(f"   f_lazy(0) = {f_lazy.evaluate(0)}")
    print(f"   f_lazy(1) = {f_lazy.evaluate(1)}")


def demo_confidence_evaluation():
    """Demonstrate confidence-based evaluation for partial functions."""
    print("\n" + "=" * 60)
    print("Confidence-Based Evaluation")
    print("=" * 60)

    print("\n1. Setup partial function with pattern")
    print("-" * 50)
    p = bf.partial(n=4)
    # Add values that suggest a pattern (e.g., parity-like)
    for i in range(8):
        p.add(i, bin(i).count("1") % 2 == 1)
    print(f"   Added {p.num_known} values (half the inputs)")

    print("\n2. Evaluate with confidence")
    print("-" * 50)
    for idx in [0, 1, 8, 9]:
        val, conf = p.evaluate_with_confidence(idx)
        status = "known" if p.is_known(idx) else "estimated"
        print(f"   p[{idx}] = {val} (confidence: {conf:.2f}, {status})")


def main():
    """Run all partial function demos."""
    print("\n" + "=" * 60)
    print("BooFun Partial Boolean Functions Demo")
    print("=" * 60)

    demo_partial_creation()
    demo_streaming()
    demo_hex_io()
    demo_storage_hints()
    demo_confidence_evaluation()

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
