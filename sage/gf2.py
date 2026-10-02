"""
Bitset linear algebra over F_2 for the parabolic computation.

Vectors are represented by nonnegative Python integers, with bit j
equal to the j-th coefficient; vector addition is bitwise XOR.

This is a .py module because SageMath's preparser interprets ^ as
exponentiation. Higher-level homological computations are implemented
in filtration.sage.
"""


def set_bits(vector):
    """Yield the indices of the nonzero bits of a nonnegative integer."""
    while vector:
        low = vector & -vector
        yield low.bit_length() - 1
        vector ^= low


def linear_image(vector, columns):
    """Apply the F_2-linear map whose basis images are stored in columns."""
    out = 0
    for j in set_bits(vector):
        out ^= columns[j]
    return out


class GF2Basis:
    """Incremental F_2 basis with highest-bit pivot reduction."""

    def __init__(self):
        self.pivots = {}

    def reduce(self, vector):
        remainder = 0
        while vector:
            pivot = vector.bit_length() - 1
            old = self.pivots.get(pivot)
            if old is None:
                bit = 1 << pivot
                remainder ^= bit
                vector ^= bit
            else:
                vector ^= old
        return remainder

    def add(self, vector):
        vector = self.reduce(vector)
        if not vector:
            return False
        self.pivots[vector.bit_length() - 1] = vector
        return True

    @property
    def rank(self):
        return len(self.pivots)


class GF2Ranker:
    """Streaming Gaussian elimination over F_2."""

    def __init__(self):
        self.pivots = {}

    def add(self, vector):
        while vector:
            pivot = vector.bit_length() - 1
            old = self.pivots.get(pivot)
            if old is None:
                self.pivots[pivot] = vector
                return True
            vector ^= old
        return False

    @property
    def rank(self):
        return len(self.pivots)


class GF2KernelTracker:
    """Streaming elimination with provenance for explicit kernel relations."""

    def __init__(self):
        self.pivots = {}

    def add_column(self, vector, index):
        witness = 1 << index
        while vector:
            pivot = vector.bit_length() - 1
            old = self.pivots.get(pivot)
            if old is None:
                self.pivots[pivot] = (vector, witness)
                return None
            vector ^= old[0]
            witness ^= old[1]
        return witness
