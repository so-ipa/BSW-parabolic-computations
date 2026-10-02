"""
Reproduce the computation DH_2(G_{5,4}; F_2) = 0.

Run from the repository root with

    sage sage/compute.sage

The script constructs G_{5,4}, verifies the cubical infrastructure,
and runs the complete support-filtration computation. No prerecorded
rank table or stream length is used as a stopping criterion.
"""

import sys
from itertools import combinations
from time import perf_counter


# ----------------------------------------------------------------------
# Load the implementation
# ----------------------------------------------------------------------

sys.path.insert(0, "sage")

load("sage/graph.sage")
load("sage/cubical.sage")
load("sage/filtration.sage")


RUN_STARTED = perf_counter()

graph = ParabolicGraph()
print("Graph:", graph.summary(), flush=True)


# ----------------------------------------------------------------------
# 1. Cubical model and structural checks
# ----------------------------------------------------------------------

print("\n=== Cubical model ===", flush=True)

cubes = CubicalModel(graph)
cubical_checks = verify_cubical_infrastructure(cubes)
print("Cubical infrastructure checks:", cubical_checks, flush=True)
inventory = cubes.inventory()
print("Square inventory:", inventory, flush=True)
checked_squares = verify_square_supports(cubes)
assert checked_squares == inventory["normalized_based_2"]
print("Verified normalized based squares:", checked_squares, flush=True)


# ----------------------------------------------------------------------
# 2. Local exact-support blocks of ranks one and two
# ----------------------------------------------------------------------

print("\n=== Rank-one local blocks ===", flush=True)

rank1 = {}

for jt in combinations(range(graph.rank), 1):
    J = frozenset(jt)

    rank1[J] = compute_low_rank_h2(cubes, J, progress_every=0)

assert all(result.h2_dimension == 0 for result in rank1.values())


print("\n=== Rank-two local blocks ===", flush=True)

rank2 = {}

for jt in combinations(range(graph.rank), 2):
    J = frozenset(jt)

    rank2[J] = compute_low_rank_h2(cubes, J, progress_every=0)

# The delta_2 construction below requires the computed local H_2 groups
# to be one-dimensional and retains one explicit representative in each.
assert all(result.h2_dimension == len(result.representatives)
           == 1 for result in rank2.values())


# ----------------------------------------------------------------------
# 3. F_1 and the connecting map delta_2
# ----------------------------------------------------------------------

print("\n=== F1 and delta2 ===", flush=True)

f1 = compute_f1(cubes)

delta2 = compute_delta2(cubes, rank1, rank2, f1)

# Rank-one relative H_2 vanishes, so the long exact sequence identifies
# H_2(F_2) with ker(delta_2).
assert delta2.h2_f2 == len(delta2.kernel)

# print("Computed dim H2(F2) =", delta2.h2_f2, flush=True)

# ----------------------------------------------------------------------
# 4. Rank-three relative blocks
# ----------------------------------------------------------------------

print("\n=== Rank-three relative blocks ===", flush=True)

rank3 = {}

for jt in combinations(range(graph.rank), 3):
    J = frozenset(jt)

    rank3[J] = prove_relative_surjectivity(cubes, J)

assert all(result.rank == len(result.block.c2) for result in rank3.values())


# ----------------------------------------------------------------------
# 5. The connecting map delta_3
# ----------------------------------------------------------------------

print("\n=== Residual coordinates and delta3 ===", flush=True)

residuals = RankTwoResiduals(cubes, rank2, delta2)

print("Global residual coordinate dimension =", residuals.dimension, flush=True)

delta3 = compute_delta3(cubes, rank3, residuals)

assert delta3.rank == delta2.h2_f2
assert len(delta3.certificates) == delta3.rank

print("rank delta3 =", delta3.rank, flush=True)

print("Independent replayed delta3 images =",
      len(delta3.certificates), flush=True)

print("Original replayable relative C3 source cycles =",
      len(delta3.source_cycles), flush=True)

# Since H_2(F_3,F_2)=0, exactness identifies H_2(F_3) with the cokernel of delta_3.
h2_f3 = delta2.h2_f2 - delta3.rank
assert h2_f3 == 0


# ----------------------------------------------------------------------
# 6. Rank-four relative blocks
# ----------------------------------------------------------------------

print("\n=== Rank-four relative blocks ===", flush=True)

rank4 = {}

for jt in combinations(range(graph.rank), 4):
    J = frozenset(jt)

    rank4[J] = prove_relative_surjectivity(cubes, J)

assert all(
    result.rank == len(result.block.c2)
    for result in rank4.values()
)

# Thus H_2(F_4,F_3)=0; together with H_2(F_3)=0, exactness gives H_2(F_4)=0.

# ----------------------------------------------------------------------
# 7. Passage from F_4 to the full cubical chain complex
# ----------------------------------------------------------------------

# Every normalized 2-cube lies in F_4. Since H_2(F_4)=0, the full normalized cubical complex also has H_2=0.
max_support_rank_c2 = max(inventory["support_distribution"])
assert max_support_rank_c2 <= 4


# ----------------------------------------------------------------------
# Final summary
# ----------------------------------------------------------------------

elapsed = perf_counter() - RUN_STARTED

print("\n=== Final verified summary ===", flush=True)
print("vertices of G_{5,4} =", len(graph.vertices), flush=True)
print(
    "normalized based 2-cubes =",
    inventory["normalized_based_2"],
    flush=True,
)
print("dim H1(F1) =", f1.h1_dimension, flush=True)
print("dim H2(F2) =", delta2.h2_f2, flush=True)
print("rank delta3 =", delta3.rank, flush=True)
print("dim H2(F3) =", h2_f3, flush=True)
print(
    "maximal support rank among normalized 2-cubes =",
    max_support_rank_c2,
    flush=True,
)
print("elapsed seconds =", round(elapsed, 3), flush=True)

print(
    "\nFINAL VERIFIED COMPUTATIONAL CONCLUSION:",
    "DH_2(G_{5,4}; F_2) = 0.",
    flush=True,
)
