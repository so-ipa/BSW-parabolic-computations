"""
Cubical-chain infrastructure for DH_2(G_{5,4}; F_2).

This module implements graph cubes, normalization by projections and
min/max connections, exact parabolic support, and enumeration of based cubes.
Optimized routines are checked against direct reference implementations.
"""

from collections import Counter
from functools import lru_cache
from itertools import zip_longest


@lru_cache(None)
def cube_edges(q):
    """Return the edges of the q-dimensional binary cube."""
    return tuple(
        (x, x | (1 << i))
        for x in range(1 << q)
        for i in range(q) if not (x >> i) & 1
    )


@lru_cache(None)
def face_indices(q, direction, epsilon):
    """Return the vertex indices of the specified codimension-one face."""
    assert q >= 1 and 0 <= direction < q and epsilon in (0, 1)
    return tuple(
        (x & ((1 << direction) - 1))
        | (epsilon << direction)
        | ((x >> direction) << (direction + 1))
        for x in range(1 << (q - 1))
    )


def face_cube(cube, q, direction, epsilon):
    """Return the specified codimension-one face."""
    return tuple(cube[j] for j in face_indices(q, direction, epsilon))


@lru_cache(None)
def earlier_neighbors(position, q):
    """Return cube neighbors preceding position in the standard ordering."""
    return tuple(
        position - (1 << d)
        for d in range(q)
        if position & (1 << d)
    )


class CubicalModel:
    """
    Graph-cube operations and the inventory of based non-degenerate 2-cubes.

    Projection- and connection-degenerate cubes vanish; no quotient is
    taken by cube symmetries or left translations.
    """

    def __init__(self, graph):
        self.graph = graph
        self.based_squares = ()
        self.square_groups = {}
        self._face_descriptors = {}

    def is_graph_map(self, cube, q):
        """Return whether cube defines a q-dimensional graph cube."""
        if len(cube) != 1 << q or any(v not in self.graph.vertices for v in cube):
            return False
        return all(cube[y] in self.graph.closed[cube[x]] for x, y in cube_edges(q))

    def normalized_reference(self, cube, q):
        """
        Direct test for projection- or connection-degeneracy.

        Used only to verify the optimized normalization routine.
        """
        for d in range(q):
            mask = 1 << d
            if all(
                cube[x] == cube[x | mask]
                for x in range(1 << q)
                if not x & mask
            ):
                return False

        for d in range(q - 1):
            for use_max in (False, True):
                fibers, good = {}, True

                for x in range(1 << q):
                    bits = [(x >> i) & 1 for i in range(q)]
                    merged = (
                        max(bits[d:d + 2])
                        if use_max
                        else min(bits[d:d + 2])
                    )
                    reduced = bits[:d] + [merged] + bits[d + 2:]
                    key = sum(b << i for i, b in enumerate(reduced))

                    if key in fibers and fibers[key] != cube[x]:
                        good = False
                        break

                    fibers[key] = cube[x]

                if good:
                    reduced_cube = tuple(
                        fibers[i]
                        for i in range(1 << (q - 1))
                    )
                    if self.is_graph_map(reduced_cube, q - 1):
                        return False

        return True

    def normalized(self, cube, q):
        """Return whether a valid q-cube survives normalization."""
        if q == 0:
            return True
        if q == 1:
            return cube[0] != cube[1]
        if q == 2:
            a, b, c, d = cube
            return not (
                (a == b and c == d)
                or (a == c and b == d)
                or a == b == c
                or b == c == d
            )
        if q == 3:
            a, b, c, d, e, f, g, h = cube
            return not (
                (a == b and c == d and e == f and g == h)
                or (a == c and b == d and e == g and f == h)
                or (a == e and b == f and c == g and d == h)
                or (a == b == c and e == f == g)
                or (b == c == d and f == g == h)
                or (a == c == e and b == d == f)
                or (c == e == g and d == f == h))
        return self.normalized_reference(cube, q)

    def translate(self, g, cube):
        """Left-translate every vertex of cube by g."""
        row = self.graph.mul[g]
        return tuple(row[v] for v in cube)

    def rebase(self, cube):
        """Translate cube so that its first vertex is the identity."""
        return self.translate(self.graph.inv[cube[0]], cube)

    def based_support(self, cube):
        """Return the exact standard parabolic support of a based cube."""
        assert cube[0] == self.graph.e
        mask = 0
        for v in cube[1:]:
            mask |= self.graph.element_mask[v]
        return frozenset(i for i in range(self.graph.rank) if mask & (1 << i))

    def support(self, cube):
        """Return the exact support of the based translate of a cube."""
        return self.based_support(self.rebase(cube))

    def face_descriptor(self, face):
        """
        Return None for a normalized-zero square, otherwise
        (initial_vertex, based_face, support).

        Square descriptors are cached because they are repeatedly queried
        while streaming 3-cubes.
        """
        if face not in self._face_descriptors:
            if not self.normalized(face, 2):
                self._face_descriptors[face] = None
            else:
                based = self.rebase(face)
                self._face_descriptors[face] = (
                    face[0],
                    based,
                    self.based_support(based),
                )
        return self._face_descriptors[face]

    def based_maps_reference(self, q, allowed=None):
        """Enumerate based q-cubes by direct backtracking."""
        if allowed is None:
            allowed = self.graph.vertices
        allowed = frozenset(allowed)
        if self.graph.e not in allowed:
            return
        assigned = [None] * (1 << q)
        assigned[0] = self.graph.e

        def recurse(position):
            if position == (1 << q):
                yield tuple(assigned)
                return
            candidates = None
            for previous in earlier_neighbors(position, q):
                nb = self.graph.closed[assigned[previous]]
                candidates = nb if candidates is None else candidates & nb
            for value in sorted(candidates & allowed):
                assigned[position] = value
                yield from recurse(position + 1)
            assigned[position] = None
        yield from recurse(1)

    def based_maps(self, q, allowed=None):
        """Enumerate based q-cubes, using the optimized dimension-three routine."""
        if q != 3:
            yield from self.based_maps_reference(q, allowed)
            return
        allowed = frozenset(
            self.graph.vertices if allowed is None else allowed)
        e = self.graph.e
        if e not in allowed:
            return
        closed = {v: self.graph.closed[v] & allowed for v in allowed}
        first = tuple(sorted(closed[e]))

        @lru_cache(None)
        def pair(a, b):
            return tuple(sorted(closed[a] & closed[b]))

        @lru_cache(None)
        def triple(a, b, c):
            return tuple(sorted(closed[a] & closed[b] & closed[c]))
        for x10 in first:
            for x01 in first:
                for x11 in pair(x10, x01):
                    for y00 in first:
                        for y10 in pair(y00, x10):
                            for y01 in pair(y00, x01):
                                for y11 in triple(y01, y10, x11):
                                    yield (e, x10, x01, x11, y00, y10, y01, y11)

    def inventory(self):
        """Build the normalized based-square inventory by Coxeter support."""
        squares, count = [], 0
        for cube in self.based_maps(2):
            count += 1
            assert self.is_graph_map(cube, 2)
            assert self.normalized(
                cube, 2) == self.normalized_reference(cube, 2)
            if self.normalized(cube, 2):
                squares.append(cube)

        self.based_squares = tuple(squares)
        assert len(set(squares)) == len(squares)

        self.square_groups = {
            J: tuple(c for c in squares if self.based_support(c) == J)
            for J in self.graph.supports
        }

        distribution = Counter(
            len(self.based_support(cube))
            for cube in squares
        )

        return {
            "all_based_2": count,
            "normalized_based_2": len(squares),
            "support_distribution": dict(sorted(distribution.items()))
        }

# ----------------------------------------------------------------------
# Structural checks for the cubical implementation
# ----------------------------------------------------------------------


def equality_patterns(length):
    """Generate canonical equality patterns of the given length."""
    if length == 0:
        yield ()
        return

    def extend(prefix, maximum):
        if len(prefix) == length:
            yield tuple(prefix)
            return

        for value in range(maximum + 2):
            yield from extend(
                prefix + [value],
                max(maximum, value),
            )

    yield from extend([0], 0)


def verify_cubical_infrastructure(cubes):
    """
    Compare optimized normalization and 3-cube enumeration with their
    direct reference implementations.
    """
    G = cubes.graph

    # Binary-cube combinatorics.
    for q in range(5):
        expected_edges = q * (1 << (q - 1)) if q else 0
        assert len(cube_edges(q)) == expected_edges

    assert face_cube(tuple(range(4)), 2, 0, 0) == (0, 2)
    assert face_cube(tuple(range(4)), 2, 1, 0) == (0, 1)

    # Projection- or connection-degenerate 3-cubes factor through a
    # 2-cube and therefore have at most four distinct vertex values.
    # We realize all relevant equality patterns inside W_{0,1}.
    clique = tuple(sorted(G.parabolics[frozenset({0, 1})]))
    labels = (
        clique
        + tuple(v for v in G.vertices if v not in clique)[:2]
    )

    normalization_count = 0

    for q in range(4):
        for pattern in equality_patterns(1 << q):
            cube = tuple(labels[i] for i in pattern)

            assert (
                cubes.normalized(cube, q)
                == cubes.normalized_reference(cube, q)
            )

            normalization_count += 1

    # Compare the optimized dimension-three enumerator with the direct
    # backtracking enumerator on the three rank-at-most-two parabolic
    # types needed here.
    generator_count = 0
    sentinel = object()

    for J in (
        frozenset({0}),
        frozenset({0, 2}),
        frozenset({0, 1}),
    ):
        WJ = G.parabolics[J]

        for left, right in zip_longest(
            cubes.based_maps(3, WJ),
            cubes.based_maps_reference(3, WJ),
            fillvalue=sentinel,
        ):
            assert left is not sentinel
            assert right is not sentinel
            assert left == right
            generator_count += 1

    return {
        "normalization_patterns": normalization_count,
        "generator_cubes": generator_count,
    }


def verify_square_supports(cubes):
    """
    Verify support minimality and face-support containment for every
    normalized based square.
    """
    G = cubes.graph

    for cube in cubes.based_squares:
        J = cubes.based_support(cube)
        image = frozenset(cube)

        for I in G.supports:
            assert (image <= G.parabolics[I]) == (J <= I)

        for d in range(2):
            for epsilon in (0, 1):
                face = face_cube(cube, 2, d, epsilon)
                assert cubes.support(face) <= J

    return len(cubes.based_squares)
