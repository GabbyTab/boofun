"""Utility modules for boolfunc package."""

from .math import (
    popcnt,
    poppar,
    over,
    subsets,
    cartesian,
    num2bin_list,
    bits,
    tensor_product,
    krawchouk,
    krawchouk2,
    hamming_distance,
    hamming_weight,
    generate_permutations,
    int_to_binary_tuple,
    binary_tuple_to_int,
)

from .number_theory import (
    gcd,
    invmod,
    crt,
    is_prime,
    prime_sieve,
)

from .finite_fields import (
    get_field,
    GFField,
    HAS_GALOIS,
)

__all__ = [
    # Math utilities
    "popcnt",
    "poppar",
    "over",
    "subsets",
    "cartesian",
    "num2bin_list",
    "bits",
    "tensor_product",
    "krawchouk",
    "krawchouk2",
    "hamming_distance",
    "hamming_weight",
    "generate_permutations",
    "int_to_binary_tuple",
    "binary_tuple_to_int",
    # Number theory
    "gcd",
    "invmod",
    "crt",
    "is_prime",
    "prime_sieve",
    # Finite fields
    "get_field",
    "GFField",
    "HAS_GALOIS",
]
