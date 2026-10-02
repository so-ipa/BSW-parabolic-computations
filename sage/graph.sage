"""
Coxeter-group and graph data for G_{5,4}.

SageMath constructs the Coxeter group W(A_5). Its elements are then
assigned integer IDs, with group operations precomputed for the
exhaustive cubical computation.
"""


# ----------------------------------------------------------------------
# Parameters
# ----------------------------------------------------------------------

CARTAN_TYPE = ["A", 5]
MAX_PARABOLIC_RANK = 2


def support_label(J):
    return tuple(i + 1 for i in sorted(J))


# ----------------------------------------------------------------------
# Coxeter graph
# ----------------------------------------------------------------------

class ParabolicGraph:
    def __init__(
        self,
        cartan_type=CARTAN_TYPE,
        max_parabolic_rank=MAX_PARABOLIC_RANK,
    ):
        self.cartan_type = cartan_type
        self.max_parabolic_rank = max_parabolic_rank

        # --------------------------------------------------------------
        # Coxeter-theoretic data
        # --------------------------------------------------------------

        self.W = CoxeterGroup(cartan_type)
        self.simple_reflections = self.W.simple_reflections()
        self.simple_indices = tuple(sorted(self.simple_reflections.keys()))
        self.rank = len(self.simple_indices)
        self.elements = tuple(self.W)
        self.element_index = {w: i for i, w in enumerate(self.elements)}
        self.vertices = tuple(range(len(self.elements)))
        self.e = self.element_index[self.W.one()]

        # --------------------------------------------------------------
        # Integer lookup tables for the exhaustive computation
        # --------------------------------------------------------------

        self.inv = tuple(self.element_index[~w] for w in self.elements)

        self.mul = tuple(
            tuple(
                self.element_index[u * v]
                for v in self.elements
            )
            for u in self.elements
        )

        self.simple = tuple(
            self.element_index[self.simple_reflections[i]]
            for i in self.simple_indices
        )

        self.supports = tuple(
            frozenset(J)
            for r in range(self.rank + 1)
            for J in Subsets(range(self.rank), r)
        )

        # --------------------------------------------------------------
        # Coxeter support
        # --------------------------------------------------------------

        # Sage labels simple reflections by 1,...,5; internally we use 0,...,4.
        self.element_support = tuple(
            frozenset(i - 1 for i in w.reduced_word())
            for w in self.elements
        )

        # Bit-mask representation used in later exhaustive computations.
        self.element_mask = tuple(
            sum(1 << j for j in J)
            for J in self.element_support
        )

        # --------------------------------------------------------------
        # Standard parabolic subgroups
        # --------------------------------------------------------------

        # Standard parabolics W_J = {w in W : supp(w) <= J}, stored by element ID.
        self.parabolics = {
            J: frozenset(
                w
                for w in self.vertices
                if self.element_support[w] <= J
            )
            for J in self.supports
        }

        # --------------------------------------------------------------
        # Definition of G_{5,4}
        # --------------------------------------------------------------

        # Relative elements defining adjacency in G_{5,4}.
        allowed = set()
        for J in self.supports:
            if len(J) <= self.max_parabolic_rank:
                allowed.update(self.parabolics[J])

        self.relative_neighbors = frozenset(allowed - {self.e})

        # Closed neighborhoods are cached for cubical backtracking.
        identity_closed = self.relative_neighbors | {self.e}

        self.closed = tuple(
            frozenset(
                self.mul[g][h]
                for h in identity_closed
            )
            for g in self.vertices
        )

        self.neighbors = tuple(
            neighborhood - {g}
            for g, neighborhood in enumerate(self.closed)
        )

        # Sage graph object used for standard graph-theoretic checks.
        self.graph = Graph(
            [
                (u, v)
                for u in self.vertices
                for v in self.neighbors[u]
                if u < v
            ]
        )

        # --------------------------------------------------------------
        # Structural checks
        # --------------------------------------------------------------

        self._check_structure()

    # ------------------------------------------------------------------
    # Cosets
    # ------------------------------------------------------------------

    def coset_representatives(self, J):
        """
        Return deterministic representatives of the left cosets of W_J.

        The smallest integer ID in each coset is chosen.
        """
        J = frozenset(J)
        WJ = self.parabolics[J]

        unused = set(self.vertices)
        representatives = []

        while unused:
            rep = min(unused)

            coset = {
                self.mul[rep][w]
                for w in WJ
            }

            representatives.append(rep)
            unused.difference_update(coset)

        assert len(representatives) * len(WJ) == len(self.vertices)

        return tuple(representatives)

    # ------------------------------------------------------------------
    # Structural verification
    # ------------------------------------------------------------------

    def _check_structure(self):
        """Check structural identities used by the later computation."""

        assert self.element_support[self.e] == frozenset()

        for j, s in enumerate(self.simple):
            assert self.element_support[s] == frozenset({j})

        # Parabolic membership via support.
        for J in self.supports:
            WJ = self.parabolics[J]

            for w in self.vertices:
                assert (w in WJ) == (self.element_support[w] <= J)

        # Standard parabolic intersections.
        for I in self.supports:
            for J in self.supports:
                assert self.parabolics[I] & self.parabolics[J] == self.parabolics[I & J]

        # Closed neighborhoods are left translates of the identity neighborhood.
        identity_closed = self.closed[self.e]

        for g in self.vertices:
            assert self.closed[g] == frozenset(
                self.mul[g][h]
                for h in identity_closed
            )

        # The adjacency relation is symmetric.
        for u in self.vertices:
            for v in self.neighbors[u]:
                assert u in self.neighbors[v]

        # Basic finite-graph checks.
        assert len(self.vertices) == self.W.cardinality()
        assert self.graph.is_connected()
        assert self.graph.is_regular()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self):
        """Return basic structural data for the graph."""
        return {
            "coxeter_type": str(self.cartan_type),
            "vertices": self.graph.num_verts(),
            "degree": self.graph.degree(self.e),
            "edges": self.graph.num_edges(),
            "closed_neighbors": len(self.closed[self.e]),
        }
