"""
Support-filtration computation for DH_2(G_{5,4}; F_2).

The homological computation is implemented in SageMath, with streaming
bitset linear algebra supplied by gf2.py. Small and medium rank
computations are cross-checked against SageMath's linear algebra over F_2.
"""

from dataclasses import dataclass
from operator import xor
from itertools import combinations

from gf2 import (
    GF2Basis,
    GF2KernelTracker,
    GF2Ranker,
    linear_image,
    set_bits,
)


# ----------------------------------------------------------------------
# Helpers for Sage cross-checks
# ----------------------------------------------------------------------

def bitset_matrix(columns, nrows):
    """Convert bitset columns to a Sage matrix over F_2."""
    columns = tuple(columns)

    return Matrix(
        GF(2),
        nrows,
        len(columns),
        lambda i, j: (columns[j] >> i) & 1,
    )


def sage_rank(columns, nrows):
    """Compute the rank of bitset columns with SageMath."""
    return bitset_matrix(columns, nrows).rank()


# ----------------------------------------------------------------------
# Exact-support chain blocks
# ----------------------------------------------------------------------

class ExactSupportBlock:
    """
    Chain block of one nonempty exact support J.

    It contains all W_J-translates of the based 1- and 2-cubes.
    Faces of smaller support vanish in the relative boundary.
    """

    def __init__(self, cubes, J):
        self.cubes = cubes
        self.graph = cubes.graph
        self.J = frozenset(J)

        assert self.J
        assert self.J <= frozenset(range(self.graph.rank))

        self.W = tuple(sorted(self.graph.parabolics[self.J]))

        self.based_c1 = tuple(
            (self.graph.e, v)
            for v in sorted(self.graph.relative_neighbors)
            if self.graph.element_support[v] == self.J
        )

        self.based_c2 = cubes.square_groups[self.J]

        self.c1 = tuple(
            (g, cube)
            for g in self.W
            for cube in self.based_c1
        )

        self.c2 = tuple(
            (g, cube)
            for g in self.W
            for cube in self.based_c2
        )

        self.c1_index = {
            basis_element: i for i,
            basis_element in enumerate(self.c1)
        }

        self.c2_index = {
            basis_element: i for i,
            basis_element in enumerate(self.c2)
        }

        assert len(self.c1_index) == len(self.c1)
        assert len(self.c2_index) == len(self.c2)

        self.d2 = tuple(
            self.boundary(g, cube, 2)
            for g, cube in self.c2
        )

    def boundary(self, g, based_cube, q):
        """
        Return the relative boundary of a translated based q-cube.

        Normalized-zero faces and faces of support strictly below J are omitted.
        """
        assert q in (2, 3)

        actual = self.cubes.translate(g, based_cube)
        index = (self.c1_index if q == 2 else self.c2_index)

        out = 0

        for d in range(q):
            for epsilon in (0, 1):
                face = face_cube(actual, q, d, epsilon)

                if not self.cubes.normalized(face, q - 1):
                    continue

                based_face = self.cubes.rebase(face)
                support = self.cubes.based_support(based_face)

                assert support <= self.J

                if support == self.J:
                    bit = 1 << index[
                        (face[0], based_face)
                    ]
                    out = xor(out, bit)

        return out

    def descriptors(self, cube):
        """Return normalized 2-face descriptors, retaining repetitions."""
        out = []

        for d in range(3):
            for epsilon in (0, 1):
                descriptor = self.cubes.face_descriptor(
                    face_cube(cube, 3, d, epsilon))

                if descriptor is not None:
                    assert descriptor[2] <= self.J
                    out.append(descriptor)

        return tuple(out)

    def d3_from_descriptors(self, g, descriptors):
        """Return the relative d_3-boundary encoded in C_2."""
        out = 0

        for h, based_face, support in descriptors:
            if support == self.J:
                bit = 1 << self.c2_index[(self.graph.mul[g][h], based_face)]
                out = xor(out, bit)

        return out


# ----------------------------------------------------------------------
# Local H_2 in ranks one and two
# ----------------------------------------------------------------------

@dataclass
class LocalH2:
    """Homology data for one exact-support block."""
    block: ExactSupportBlock
    rank_d2: int
    image_d3: GF2Basis
    representatives: tuple
    certificates: tuple

    @property
    def h2_dimension(self):
        """
        Dimension of H_2 of the exact-support block.
        """
        return (len(self.block.c2) - self.rank_d2 - self.image_d3.rank)


def compute_low_rank_h2(
    cubes,
    J,
    progress_every=100000,
    sage_cross_check=True,
):
    """
    Compute H_2 of a rank-one or rank-two exact-support block.

    All normalized based 3-cubes of exact support J and their W_J-translates
    are processed exhaustively. When requested, d_2 and d_3 ranks are
    independently cross-checked with SageMath.
    """
    block = ExactSupportBlock(cubes, J)

    assert len(block.J) in (1, 2)

    # Kernel and rank of d_2

    kernel_tracker = GF2KernelTracker()
    kernel = []

    for j, column in enumerate(block.d2):
        relation = kernel_tracker.add_column(column, j)

        if relation is not None:
            kernel.append(relation)

    rank_d2 = len(block.c2) - len(kernel)

    # Image of d_3

    image = GF2Basis()
    certificates = []

    based_c3_count = 0
    column_count = 0

    for cube in cubes.based_maps(3, block.W):
        if not cubes.normalized(cube, 3):
            continue

        if cubes.based_support(cube) != block.J:
            continue

        based_c3_count += 1
        descriptors = block.descriptors(cube)

        for g in block.W:
            column = block.d3_from_descriptors(g, descriptors)

            column_count += 1

            # d_2 d_3 = 0.
            assert linear_image(column, block.d2) == 0

            if image.add(column):
                certificates.append((column, g, cube))

        if (progress_every and based_c3_count % progress_every == 0):
            print(
                "support",
                support_label(J),
                "based C3",
                based_c3_count,
                "rank d3",
                image.rank,
                flush=True,
            )

    # Replay every independent d_3 certificate directly from the cube.
    for column, g, cube in certificates:
        assert column == block.boundary(g, cube, 3)

    # Quotient ker(d_2) / im(d_3)

    quotient = GF2Basis()
    representatives = []

    for cycle in kernel:
        assert linear_image(cycle, block.d2) == 0

        reduced_cycle = image.reduce(cycle)

        if quotient.add(reduced_cycle):
            representatives.append(cycle)

    assert (len(representatives) == len(kernel) - image.rank)

    # Independent SageMath rank cross-check

    if sage_cross_check:
        sage_rank_d2 = sage_rank(block.d2, len(block.c1))
        assert sage_rank_d2 == rank_d2

        certificate_columns = tuple(column for column, _, _ in certificates)
        sage_rank_d3 = sage_rank(certificate_columns, len(block.c2))
        assert sage_rank_d3 == image.rank

    result = LocalH2(
        block=block,
        rank_d2=rank_d2,
        image_d3=image,
        representatives=tuple(representatives),
        certificates=tuple(certificates),
    )

    print(
        "J =",
        support_label(J),
        "|W_J| =",
        len(block.W),
        "C2 =",
        len(block.c2),
        "rank d2 =",
        rank_d2,
        "rank d3 =",
        image.rank,
        "H2 =",
        result.h2_dimension,
        "exhaustive columns =",
        column_count,
        flush=True,
    )

    return result


# ----------------------------------------------------------------------
# The first filtration stage F_1
# ----------------------------------------------------------------------

@dataclass
class F1Data:
    """
    Chain and homology data for F_1.
    """
    c1: tuple
    c2: tuple
    c1_index: dict
    d1: tuple
    d2: tuple
    image_d2: GF2Basis
    rank_d1: int

    @property
    def h1_dimension(self):
        """
        Dimension of H_1(F_1).
        """
        return (
            len(self.c1)
            - self.rank_d1
            - self.image_d2.rank
        )


def compute_f1(cubes, sage_cross_check=True):
    """
    Construct the global chain complex in degrees 2, 1, and 0 for F_1.

    The boundary maps d_1 and d_2 are computed explicitly, and
    d_1 d_2 = 0 is checked for every normalized 2-cube.

    If sage_cross_check is True, the ranks of d_1 and d_2 are also
    recomputed using SageMath matrices over F_2.
    """
    G = cubes.graph

    based_c1 = tuple(
        (G.e, v)
        for v in sorted(G.relative_neighbors)
        if len(G.element_support[v]) == 1
    )

    based_c2 = tuple(
        cube
        for cube in cubes.based_squares
        if len(cubes.based_support(cube)) == 1
    )

    c1 = tuple(
        cubes.translate(g, cube)
        for g in G.vertices
        for cube in based_c1
    )

    c2 = tuple(
        cubes.translate(g, cube)
        for g in G.vertices
        for cube in based_c2
    )

    c1_index = {
        cube: i
        for i, cube in enumerate(c1)
    }

    assert len(c1_index) == len(c1)
    assert len(set(c2)) == len(c2)

    # d_1 sends an oriented edge (u,v) to u+v over F_2.
    d1 = tuple(
        xor(1 << u, 1 << v)
        for u, v in c1
    )

    d2 = []
    image_d2 = GF2Basis()

    for cube in c2:
        column = 0

        for d in range(2):
            for epsilon in (0, 1):
                face = face_cube(
                    cube,
                    2,
                    d,
                    epsilon,
                )

                if cubes.normalized(face, 1):
                    assert len(cubes.support(face)) == 1

                    bit = 1 << c1_index[face]
                    column = xor(column, bit)

        assert linear_image(column, d1) == 0

        image_d2.add(column)
        d2.append(column)

    ranker_d1 = GF2Ranker()

    for column in d1:
        ranker_d1.add(column)

    if sage_cross_check:
        assert sage_rank(
            d1,
            len(G.vertices),
        ) == ranker_d1.rank

        assert sage_rank(
            d2,
            len(c1),
        ) == image_d2.rank

    result = F1Data(
        c1=tuple(c1),
        c2=tuple(c2),
        c1_index=c1_index,
        d1=tuple(d1),
        d2=tuple(d2),
        image_d2=image_d2,
        rank_d1=ranker_d1.rank,
    )

    print(
        "F1:",
        "C1 =", len(result.c1),
        "C2 =", len(result.c2),
        "rank d1 =", result.rank_d1,
        "rank d2 =", result.image_d2.rank,
        "H1 =", result.h1_dimension,
        flush=True,
    )

    return result


# ----------------------------------------------------------------------
# Direct normalized boundaries and the connecting map delta_2
# ----------------------------------------------------------------------

def full_normalized_boundary(
    cubes,
    chain,
    basis,
    q,
    translation=None,
):
    """
    Input:
        A q-chain encoded as a bitset in a basis of translated based cubes,
        with an optional additional left translation.
    Output:
        Its full normalized boundary as the set of actual (q-1)-cubes
        having coefficient one over F_2.

    No projection by support is applied.  The direct reference
    normalization is used deliberately for certificate replay.
    """
    out = set()

    if translation is None:
        translation = cubes.graph.e

    for j in set_bits(chain):
        g, cube = basis[j]

        actual = cubes.translate(
            cubes.graph.mul[translation][g],
            cube,
        )

        for d in range(q):
            for epsilon in (0, 1):
                face = face_cube(
                    actual,
                    q,
                    d,
                    epsilon,
                )

                if cubes.normalized_reference(
                    face,
                    q - 1,
                ):
                    if face in out:
                        out.remove(face)
                    else:
                        out.add(face)

    return out


def connecting_boundary(
    cubes,
    local,
    f1,
    representative,
):
    """
    Compute the connecting boundary of the translated local H_2 generator
    of a rank-two exact-support block.

    The result is encoded as a bitset in C_1(F_1).
    """
    assert len(local.representatives) == 1

    boundary = full_normalized_boundary(
        cubes,
        local.representatives[0],
        local.block.c2,
        2,
        representative,
    )

    out = 0

    for edge in boundary:
        assert len(cubes.support(edge)) == 1

        bit = 1 << f1.c1_index[edge]
        out = xor(out, bit)

    assert linear_image(out, f1.d1) == 0

    return out


@dataclass
class Delta2Data:
    """
    Computed data for the connecting map delta_2.
    """
    labels: tuple
    raw_columns: tuple
    columns: tuple
    kernel: tuple
    table: tuple
    coset_reps: dict
    rank: int

    @property
    def h2_f2(self):
        """
        Dimension of H_2(F_2), identified with ker(delta_2) after the
        previously verified vanishing of the rank-one relative H_2 groups.
        """
        return len(self.columns) - self.rank


def compute_delta2(
    cubes,
    rank1,
    rank2,
    f1,
    sage_cross_check=True,
):
    """
    Compute the connecting map

        delta_2 : H_2(F_2,F_1) -> H_1(F_1).

    Every rank-two support and every left coset is processed.  Kernel
    relations are retained explicitly and replayed.

    If sage_cross_check is True, the rank of the induced map modulo
    im(d_2 : C_2(F_1) -> C_1(F_1)) is independently recomputed with
    SageMath matrices over F_2.
    """
    assert all(
        result.h2_dimension == 0
        for result in rank1.values()
    )

    G = cubes.graph

    image = GF2Basis()
    tracker = GF2KernelTracker()

    labels = []
    raw_columns = []
    columns = []
    kernel = []
    table = []
    coset_reps = {}

    domain_dimension = 0

    for jt in combinations(
        range(G.rank),
        2,
    ):
        J = frozenset(jt)
        local = rank2[J]

        assert (
            local.h2_dimension
            == len(local.representatives)
            == 1
        )

        reps = G.coset_representatives(J)
        coset_reps[J] = reps

        domain_dimension += (
            len(reps)
            * local.h2_dimension
        )

        before = image.rank

        for rep in reps:
            boundary = connecting_boundary(
                cubes,
                local,
                f1,
                rep,
            )

            # Work in H_1(F_1) by reducing modulo im d_2.
            reduced = f1.image_d2.reduce(boundary)

            assert linear_image(
                reduced,
                f1.d1,
            ) == 0

            relation = tracker.add_column(
                reduced,
                len(columns),
            )

            if relation is not None:
                kernel.append(relation)

            image.add(reduced)

            labels.append((J, rep))
            raw_columns.append(boundary)
            columns.append(reduced)

        table.append(
            (
                support_label(J),
                len(reps),
                image.rank - before,
                image.rank,
            )
        )

        print(
            "J =", support_label(J),
            "cosets =", len(reps),
            "new rank =", image.rank - before,
            "cumulative =", image.rank,
            flush=True,
        )

    assert domain_dimension == len(columns)

    # --------------------------------------------------------------
    # Direct rank check without quotient coordinates
    # --------------------------------------------------------------

    direct_ranker = GF2Ranker()

    for column in f1.image_d2.pivots.values():
        direct_ranker.add(column)

    boundary_rank = direct_ranker.rank

    for column in raw_columns:
        direct_ranker.add(column)

    assert (
        direct_ranker.rank - boundary_rank
        == image.rank
    )

    # --------------------------------------------------------------
    # Replay and independence of kernel witnesses
    # --------------------------------------------------------------

    for relation in kernel:
        assert relation
        assert linear_image(
            relation,
            columns,
        ) == 0

    kernel_ranker = GF2Ranker()

    for relation in kernel:
        kernel_ranker.add(relation)

    assert kernel_ranker.rank == len(kernel)

    assert (
        len(kernel)
        == domain_dimension - image.rank
    )

    # --------------------------------------------------------------
    # Independent SageMath rank cross-check
    # --------------------------------------------------------------

    if sage_cross_check:
        # rank(im d_2)
        sage_boundary_rank = sage_rank(
            f1.d2,
            len(f1.c1),
        )

        assert (
            sage_boundary_rank
            == f1.image_d2.rank
        )

        # rank(im d_2 + raw connecting boundaries)
        combined_columns = (
            tuple(f1.d2)
            + tuple(raw_columns)
        )

        sage_combined_rank = sage_rank(
            combined_columns,
            len(f1.c1),
        )

        assert (
            sage_combined_rank
            - sage_boundary_rank
            == image.rank
        )

        # The reduced columns themselves must have the same rank.
        assert sage_rank(
            columns,
            len(f1.c1),
        ) == image.rank

    result = Delta2Data(
        labels=tuple(labels),
        raw_columns=tuple(raw_columns),
        columns=tuple(columns),
        kernel=tuple(kernel),
        table=tuple(table),
        coset_reps=coset_reps,
        rank=image.rank,
    )

    print(
        "delta2:",
        "domain =", len(result.columns),
        "rank =", result.rank,
        "kernel =", len(result.kernel),
        "dim H2(F2) =", result.h2_f2,
        flush=True,
    )

    return result


# ----------------------------------------------------------------------
# Relative H_2 vanishing in ranks three and four
# ----------------------------------------------------------------------

@dataclass
class RelativeSurjectivity:
    """
    Certificate data proving surjectivity of the relative boundary map d_3
    for one exact-support block of rank three or four.
    """
    block: ExactSupportBlock
    rank: int
    certificates: tuple


def prism_3cubes_over_square(cubes, square, J):
    """
    Enumerate all normalized based 3-cubes whose bottom face is the given
    normalized based square and whose vertices lie in W_J.

    The bottom square is fixed.  The top vertices are constructed by
    intersecting the required closed neighborhoods, so every yielded
    tuple is a graph map before normalization is tested.
    """
    G = cubes.graph
    J = frozenset(J)
    allowed = G.parabolics[J]

    x00, x10, x01, x11 = square

    for y00 in sorted(
        G.closed[x00] & allowed
    ):
        for y10 in sorted(
            G.closed[x10]
            & G.closed[y00]
            & allowed
        ):
            for y01 in sorted(
                G.closed[x01]
                & G.closed[y00]
                & allowed
            ):
                for y11 in sorted(
                    G.closed[x11]
                    & G.closed[y10]
                    & G.closed[y01]
                    & allowed
                ):
                    cube = (
                        x00, x10, x01, x11,
                        y00, y10, y01, y11,
                    )

                    if cubes.normalized(cube, 3):
                        assert (
                            cubes.based_support(cube)
                            == J
                        )
                        yield cube


def prove_relative_surjectivity(cubes, J):
    """
    Prove that the relative boundary map d_3 is surjective for one
    exact-support block of rank three or four.

    Constant-top cones are processed first.  If they do not already span
    C_2, general prisms are processed next.  The search stops only when
    the image has full C_2 rank; no previously recorded homology value is
    used as a stopping criterion.

    Every independent column retained in the rank certificate is replayed
    directly from its original 3-cube.
    """
    block = ExactSupportBlock(cubes, J)

    assert len(block.J) in (3, 4)
    assert not block.c1
    assert not any(block.d2)

    image = GF2Ranker()
    certificates = []

    cones = 0
    prisms = 0
    columns = 0

    target = len(block.c2)

    def insert(cube, kind, square_index):
        nonlocal columns

        descriptors = block.descriptors(cube)

        for g in block.W:
            column = block.d3_from_descriptors(
                g,
                descriptors,
            )

            columns += 1

            if image.add(column):
                certificates.append(
                    (
                        column,
                        kind,
                        square_index,
                        cube[4:],
                        g,
                    )
                )

    # --------------------------------------------------------------
    # Constant-top cones
    # --------------------------------------------------------------

    for j, square in enumerate(
        block.based_c2
    ):
        apices = set(block.W)

        for vertex in square:
            apices.intersection_update(
                cubes.graph.closed[vertex]
            )

        for apex in sorted(apices):
            cube = square + (apex,) * 4

            if not cubes.normalized(cube, 3):
                continue

            cones += 1
            insert(
                cube,
                "cone",
                j,
            )

            if image.rank == target:
                break

        if image.rank == target:
            break

    # --------------------------------------------------------------
    # General prisms, only if cones were insufficient
    # --------------------------------------------------------------

    if image.rank < target:
        for j, square in enumerate(
            block.based_c2
        ):
            for cube in prism_3cubes_over_square(
                cubes,
                square,
                block.J,
            ):
                # Constant-top cubes were already processed above.
                if (
                    cube[4]
                    == cube[5]
                    == cube[6]
                    == cube[7]
                ):
                    continue

                prisms += 1
                insert(
                    cube,
                    "prism",
                    j,
                )

                if image.rank == target:
                    break

            if image.rank == target:
                break

    assert (
        image.rank
        == len(certificates)
        == target
    ), (
        "surjectivity not established",
        support_label(J),
    )

    # --------------------------------------------------------------
    # Replay every independent certificate directly
    # --------------------------------------------------------------

    replay_ranker = GF2Ranker()

    for (
        column,
        kind,
        square_index,
        top,
        g,
    ) in certificates:
        cube = (
            block.based_c2[square_index]
            + top
        )

        assert cubes.is_graph_map(cube, 3)
        assert cubes.normalized_reference(
            cube,
            3,
        )
        assert (
            cubes.based_support(cube)
            == block.J
        )

        assert column == block.boundary(
            g,
            cube,
            3,
        )

        replay_ranker.add(column)

    assert replay_ranker.rank == target

    result = RelativeSurjectivity(
        block=block,
        rank=image.rank,
        certificates=tuple(certificates),
    )

    print(
        "J =", support_label(J),
        "C2 =", target,
        "rank d3 =", image.rank,
        "cones =", cones,
        "prisms =", prisms,
        "columns =", columns,
        "replayed =", len(certificates),
        flush=True,
    )

    return result


# ----------------------------------------------------------------------
# Rank-two residual coordinates and the connecting map delta_3
# ----------------------------------------------------------------------

class RankTwoResiduals:
    """
    Input:
        The cubical model, the local rank-two homology data, and the computed data for delta_2.
    State:
        Coordinates on the direct sum of the quotients C_2^{g,J} / im(d_3^{g,J}) over all rank-two supports J and cosets gW_J.

    These residual coordinates are auxiliary and are not coordinates on H_2(F_2).
    """

    def __init__(self, cubes, rank2, delta2):
        self.cubes, self.graph, self.delta2 = cubes, cubes.graph, delta2
        self.labels = delta2.labels
        self.index = {label: i for i, label in enumerate(self.labels)}
        self.reducers, self.vertex_to_rep, self.blocks = {}, {}, {}
        total = 0
        for jt in combinations(range(self.graph.rank), 2):
            J = frozenset(jt)
            local = rank2[J]
            image = local.image_d3
            free = tuple(j for j in range(len(local.block.c2))
                         if j not in image.pivots)
            position = {j: i for i, j in enumerate(free)}

            def compact(v):
                return sum(1 << position[j] for j in set_bits(v))
            units = tuple(compact(image.reduce(1 << j))
                          for j in range(len(local.block.c2)))
            seed = compact(image.reduce(local.representatives[0]))
            assert seed and len(free) == local.rank_d2 + local.h2_dimension
            for col, _, _ in local.certificates:
                assert linear_image(col, units) == 0
            self.reducers[J] = dict(
                block=local.block, width=len(free), units=units, seed=seed)
            vmap = {}
            for rep in delta2.coset_reps[J]:
                label = (J, rep)
                for w in local.block.W:
                    v = self.graph.mul[rep][w]
                    assert v not in vmap
                    vmap[v] = rep
                self.blocks[label] = dict(start=total, width=len(free), seed=seed,
                                          class_index=self.index[label])
                total += len(free)
            assert len(vmap) == len(self.graph.vertices)
            self.vertex_to_rep[J] = vmap
        self.dimension = total

    def encode_face(self, h, bf, J):
        """
        Input:
            A normalized rank-two face represented by its initial vertex h, its based translate bf, and its exact support J.
        Output:
            Its coordinate in the global rank-two residual space.
        """
        rep = self.vertex_to_rep[J][h]
        lg = self.graph.mul[self.graph.inv[rep]][h]
        R = self.reducers[J]
        j = R['block'].c2_index[(lg, bf)]
        return R['units'][j] << self.blocks[(J, rep)]['start']

    def decode_cycle(self, vector):
        """
        Input:
            A residual vector representing a relative 2-cycle.
        Output:
            Its coordinate in the direct sum of the local rank-two H_2 groups.

        Each local component must be either zero or the nonzero local H_2 generator.
        """
        out = 0
        assert vector >= 0 and vector.bit_length() <= self.dimension
        for label in self.labels:
            B = self.blocks[label]
            piece = (vector >> B['start']) & ((1 << B['width']) - 1)
            assert piece in (
                0, B['seed']), ('not a local H2 class', label, piece)
            if piece:
                out = xor(out, 1 << B['class_index'])
        return out

    def translate_class(self, vector, r):
        """
        Input:
            A class in the direct sum of the local rank-two H_2 groups and a group element r.
        Output:
            The class obtained by left translation by r.
        """
        out = 0
        for i in set_bits(vector):
            J, rep = self.labels[i]
            new_rep = self.vertex_to_rep[J][self.graph.mul[r][rep]]
            out = xor(out, 1 << self.index[(J, new_rep)])
        return out

    def encode_boundary(self, faces):
        """
        Input:
            A normalized 2-chain represented by the set of its actual faces.
        Output:
            Its rank-two component encoded in the global residual space.
        """
        out = 0
        for f in faces:
            bf = self.cubes.rebase(f)
            J = self.cubes.based_support(bf)
            assert len(J) <= 2
            if len(J) == 2:
                out = xor(out, self.encode_face(f[0], bf, J))
        return out


class DepTracker:
    """
    Input:
        Pairs (a, b), where a is a rank-three relative boundary and b is the corresponding rank-two residual.
    State:
        Gaussian-elimination data for detecting linear dependencies among the upper vectors a, together with the corresponding lower residuals and source witnesses.
    """

    def __init__(self):
        self.pivots = {}
        self.raw_sources = []
        self.unit_lower = None
        self.unit_witness = None

    def add(self, a, b, source):
        """
        Input:
            An upper vector a, the corresponding lower residual b, and its source.
        Output:
            None if a is independent of the stored upper vectors; otherwise, the resulting lower residual and a witness for the dependency.
        """
        witness = 0
        while a:
            k = a.bit_length() - 1
            if k not in self.pivots:
                witness = xor(witness, 1 << len(self.raw_sources))
                self.raw_sources.append(source)
                self.pivots[k] = (a, b, witness)
                return None
            old_a, old_b, old_witness = self.pivots[k]
            a = xor(a, old_a)
            b = xor(b, old_b)
            witness = xor(witness, old_witness)
        return b, witness

    @property
    def rank(self):
        """
         Output:
            The dimension of the span of the stored upper vectors.
        """
        return len(self.pivots)

    def freeze(self, dimension):
        """
        Input:
            The full dimension of the upper space.
        State:
            Lookup tables for eliminating upper vectors and tracking the
            corresponding lower residuals and source witnesses.

        This method is called only after the stored upper vectors have full rank.
        """
        assert set(self.pivots) == set(range(dimension))
        lower, witness = [], []
        for k in range(dimension):
            a, b, w = self.pivots[k]
            a = xor(a, 1 << k)
            lower.append(xor(b, linear_image(a, lower)))
            witness.append(xor(w, linear_image(a, witness)))
        self.unit_lower, self.unit_witness = tuple(lower), tuple(witness)
        # Check lookup evaluation against ordinary pivot elimination on every unit.
        for k in range(dimension):
            a, b = 1 << k, 0
            while a:
                p = a.bit_length() - 1
                a = xor(a, self.pivots[p][0])
                b = xor(b, self.pivots[p][1])
            assert b == self.unit_lower[k]


class LocalResidualView:
    """
    Input:
        The global rank-two residual coordinates and a rank-three support J.
    State:
        A lossless packing of the residual blocks that can occur inside W_J.

    This removes only coordinates that are identically zero in the local computation; it does not take any additional chain or homology quotient.
    """

    def __init__(self, residuals, J):
        self.residuals, self.J = residuals, frozenset(J)
        self.W = residuals.graph.parabolics[self.J]
        self.blocks = {}
        self.dimension = 0
        for label in residuals.labels:
            I, rep = label
            if I <= self.J and rep in self.W:
                old = residuals.blocks[label]
                self.blocks[label] = dict(start=self.dimension, width=old['width'],
                                          seed=old['seed'], class_index=old['class_index'])
                self.dimension += old['width']

    def encode_face(self, h, bf, I):
        """
        Input:
            A normalized rank-two face in W_J, represented by its initial vertex h, its based translate bf, and its exact support I.
        Output:
            Its coordinate in the packed local residual space.
        """
        R = self.residuals.reducers[I]
        rep = self.residuals.vertex_to_rep[I][h]
        lg = self.residuals.graph.mul[self.residuals.graph.inv[rep]][h]
        index = R['block'].c2_index[(lg, bf)]
        return R['units'][index] << self.blocks[(I, rep)]['start']

    def embed(self, vector):
        """
        Input:
            A vector in the packed local residual coordinates.
        Output:
            The same vector embedded in the global rank-two residual space.
        """
        out = 0
        assert vector >= 0 and vector.bit_length() <= self.dimension
        for label, B in self.blocks.items():
            piece = (vector >> B['start']) & ((1 << B['width']) - 1)
            out = xor(out, piece << self.residuals.blocks[label]['start'])
        return out

    def decode(self, vector):
        """
        Input:
            A packed residual vector representing a relative 2-cycle.
        Output:
            Its coordinate in the direct sum of the local rank-two H_2 groups.
        """
        out = 0
        assert vector >= 0 and vector.bit_length() <= self.dimension
        for label, B in self.blocks.items():
            piece = (vector >> B['start']) & ((1 << B['width']) - 1)
            assert piece in (0, B['seed']), ('not a cycle', label, piece)
            if piece:
                out = xor(out, 1 << B['class_index'])
        return out


@dataclass
class Delta3SourceCycle:
    """
    Stores a replayable relative 3-cycle and its image under delta_3.
    """
    support: frozenset
    candidate_number: int
    terms: tuple
    class_vector: int


@dataclass
class Delta3Data:
    """
    Stores the computed data and certificates for the connecting map delta_3.
    """
    rank: int
    table: tuple
    source_cycles: tuple
    certificates: tuple


def verify_delta3_source(
    cubes, residuals, view, tracker,
    source, witness, dependency, candidate_number
):
    """
    Input:
        A detected upper dependency together with its source provenance.
    Output:
        A replayed relative 3-cycle and its class under delta_3.

    The actual 3-chain is reconstructed and its full boundary is recomputed independently of the paired elimination.
    """
    terms = tuple(tracker.raw_sources[i]
                  for i in set_bits(witness)) + (source,)
    for g, cube in terms:
        assert g in view.W and cubes.is_graph_map(cube, 3)
        assert cubes.normalized_reference(cube, 3)
        assert cubes.based_support(cube) == view.J
    faces = full_normalized_boundary(cubes, (1 << len(terms)) - 1, terms, 3)
    assert all(len(cubes.support(f)) <= 2 for f in faces)
    replay_residual = residuals.encode_boundary(faces)
    assert replay_residual == view.embed(dependency)
    q = residuals.decode_cycle(replay_residual)
    assert q == view.decode(dependency)
    assert linear_image(q, residuals.delta2.columns) == 0
    return Delta3SourceCycle(view.J, candidate_number, terms, q)


def compute_delta3(cubes, rank3_relative, residuals, progress_every=0):
    """
    Input:
        The cubical model, the rank-three relative computations, and the
        rank-two residual coordinates.
    Output:
        The computed data and certificates for the connecting map delta_3.

    Rank-three relative cycles are detected by paired elimination, and their
    boundaries are decoded as classes in H_2(F_2). The computation stops only
    when these classes span H_2(F_2).
    """
    G = cubes.graph
    image = GF2Ranker()
    table = []
    source_cycles = []
    certs = []

    target = residuals.delta2.h2_f2
    stop = False

    face_maps = tuple(
        face_indices(3, d, epsilon)
        for d in range(3)
        for epsilon in (0, 1)
    )

    for jt in combinations(range(G.rank), 3):
        if stop:
            break

        J = frozenset(jt)
        block = rank3_relative[J].block

        assert rank3_relative[J].rank == len(block.c2)

        view = LocalResidualView(residuals, J)
        tracker = DepTracker()
        reps = G.coset_representatives(J)

        before = image.rank
        nc = nd = columns = 0
        seen_dependencies = {0}
        effect_cache = {}
        frozen = False

        def column_pair(g, descriptors):
            a = b = 0

            for h, bf, I in descriptors:
                gh = G.mul[g][h]

                if I == J:
                    a = xor(a, 1 << block.c2_index[(gh, bf)])
                elif len(I) == 2:
                    b = xor(b, view.encode_face(gh, bf, I))
                else:
                    assert I < J and len(I) <= 1

            return a, b

        def frozen_effect(desc):
            if desc not in effect_cache:
                h, bf, I = desc

                if I == J:
                    effect_cache[desc] = tuple(
                        tracker.unit_lower[
                            block.c2_index[(G.mul[g][h], bf)]
                        ]
                        for g in block.W
                    )

                elif len(I) == 2:
                    effect_cache[desc] = tuple(
                        view.encode_face(G.mul[g][h], bf, I)
                        for g in block.W
                    )

                else:
                    assert I < J and len(I) <= 1
                    effect_cache[desc] = (0,) * len(block.W)

            return effect_cache[desc]

        def process_dependency(dep, g, bc, descriptors, witness=None):
            nonlocal stop

            if dep in seen_dependencies:
                return

            seen_dependencies.add(dep)

            q = view.decode(dep)

            assert q == residuals.decode_cycle(view.embed(dep))
            assert linear_image(q, residuals.delta2.columns) == 0

            if not q:
                return

            pending = []

            for r in reps:
                tq = residuals.translate_class(q, r)

                assert linear_image(tq, residuals.delta2.columns) == 0

                if image.add(tq):
                    pending.append((r, tq))

                assert image.rank <= target

                if image.rank == target:
                    stop = True
                    break

            if not pending:
                return

            if witness is None:
                a, b = column_pair(g, descriptors)

                assert dep == xor(
                    b,
                    linear_image(a, tracker.unit_lower),
                )

                witness = linear_image(
                    a, tracker.unit_witness
                )

            original = verify_delta3_source(
                cubes,
                residuals,
                view,
                tracker,
                (g, bc),
                witness,
                dep,
                nc,
            )

            source_index = len(source_cycles)
            source_cycles.append(original)

            for r, tq in pending:
                full = full_normalized_boundary(
                    cubes,
                    (1 << len(original.terms)) - 1,
                    original.terms,
                    3,
                    r,
                )

                assert all(
                    len(cubes.support(f)) <= 2
                    for f in full
                )

                assert residuals.decode_cycle(
                    residuals.encode_boundary(full)
                ) == tq

                certs.append((source_index, r, tq))

        for bc in cubes.based_maps(3, block.W):
            if (
                not cubes.normalized(bc, 3)
                or cubes.based_support(bc) != J
            ):
                continue

            nc += 1
            descriptors = []

            for indices in face_maps:
                face = tuple(bc[i] for i in indices)
                desc = cubes.face_descriptor(face)

                if desc is not None:
                    assert desc[2] <= J
                    descriptors.append(desc)

            descriptors = tuple(descriptors)

            if not frozen:
                for g in block.W:
                    a, b = column_pair(g, descriptors)
                    result = tracker.add(a, b, (g, bc))
                    columns += 1

                    if result is not None:
                        nd += 1
                        dep, witness = result

                        process_dependency(
                            dep,
                            g,
                            bc,
                            descriptors,
                            witness,
                        )

                    if stop:
                        break

                if (
                    not stop
                    and tracker.rank == len(block.c2)
                ):
                    tracker.freeze(len(block.c2))
                    frozen = True

            else:
                effects = [
                    frozen_effect(desc)
                    for desc in descriptors
                ]

                dependencies = [0] * len(block.W)

                for row in effects:
                    dependencies = [
                        xor(a, b)
                        for a, b in zip(dependencies, row)
                    ]

                for g, dep in zip(block.W, dependencies):
                    columns += 1
                    nd += 1

                    process_dependency(
                        dep,
                        g,
                        bc,
                        descriptors,
                    )

                    if stop:
                        break

            if stop:
                break

            if progress_every and nc % progress_every == 0:
                print(
                    'J =', support_label(J),
                    'candidates =', nc,
                    'upper rank =', tracker.rank,
                    'image rank =', image.rank, '/', target,
                    flush=True,
                )

        table.append(
            dict(
                J=support_label(J),
                C2=len(block.c2),
                candidates=nc,
                columns=columns,
                upper_rank=tracker.rank,
                dependencies=nd,
                new_rank=image.rank - before,
                cumulative_rank=image.rank,
                # exhausted=not stop,
            )
        )

        print(
            'completed J =', support_label(J),
            'columns =', columns,
            'upper rank =', tracker.rank,
            'new rank =', image.rank - before,
            'cumulative rank =', image.rank,
            # 'exhausted =', not stop,
            flush=True,
        )

    assert image.rank == target, (
        'Selected connecting images did not span H2(F2).'
    )

    certificate_ranker = GF2Ranker()

    for _, _, q in certs:
        certificate_ranker.add(q)

    assert len(certs) == image.rank
    assert certificate_ranker.rank == image.rank

    return Delta3Data(
        image.rank,
        tuple(table),
        tuple(source_cycles),
        tuple(certs),
    )
