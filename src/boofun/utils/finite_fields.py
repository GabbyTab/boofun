"""Optional finite field helpers (thin wrapper around ``galois`` if available)."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["get_field", "GFField", "HAS_GALOIS"]

try:  # pragma: no cover - optional dependency
    import galois as _galois

    HAS_GALOIS = True
except Exception:  # pragma: no cover
    _galois = None
    HAS_GALOIS = False


@dataclass(frozen=True)
class GFField:
    """Descriptor for GF(p^m)."""

    p: int
    m: int = 1

    @property
    def order(self) -> int:
        return self.p**self.m

    def element_type(self):  # pragma: no cover - runtime dispatch
        if HAS_GALOIS:
            return _galois.GF(self.order)
        if self.p == 2 and self.m == 1:
            return int  # GF(2) fallback using plain ints
        raise ImportError("galois is required for GF(p^m) when m>1")


def get_field(p: int = 2, m: int = 1) -> GFField:
    """Return a simple GF(p^m) descriptor."""

    return GFField(p=p, m=m)


