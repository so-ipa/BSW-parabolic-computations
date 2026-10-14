# Computation of DH_2(G_{5,4}; F_2)

This repository contains the SageMath code used to reproduce the computation

    DH_2(G_{5,4}; F_2) = 0.

The computation uses a support-rank filtration (defined using based translates and standard parabolic subgroups) and performs the required
cubical enumeration and linear algebra directly.

## Requirements

The computation has been tested with SageMath 10.8.

## Running the computation

From the repository root, run:

```bash
sage sage/compute.sage
```

To display the output and save it simultaneously, run:

```bash
sage sage/compute.sage | tee results/compute_output.txt
```

A complete reference run is included in `results/compute_output.txt`.

## Repository structure

```text
parabolic/
├── README.md
├── LICENSE
├── sage/
│   ├── graph.sage
│   ├── cubical.sage
│   ├── gf2.py
│   ├── filtration.sage
│   └── compute.sage
└── results/
    └── compute_output.txt
```

### `sage/graph.sage`

Constructs the Coxeter group of type A_5 and the graph G_{5,4}. Group elements
are assigned integer IDs and the group operations needed by the exhaustive
computation are stored as lookup tables.

### `sage/cubical.sage`

Implements the cubical-chain infrastructure used in the computation, including
graph cubes, cubical faces, normalization, exact parabolic support, and the enumeration
of based cubes. It also contains direct reference implementations used to check
the optimized routines.

### `sage/gf2.py`

Implements low-level bitset linear algebra over F_2, including streaming
Gaussian elimination and kernel-relation tracking.

### `sage/filtration.sage`

Implements the support-rank filtration computation: local exact-support blocks, the
first filtration stage, the connecting homomorphisms, and the relative surjectivity
calculations used in the proof.

### `sage/compute.sage`

Main reproducibility script. It constructs G_{5,4}, verifies the cubical
infrastructure, runs the complete support-rank filtration computation, and checks
the deductions leading to the final vanishing result.

### `results/compute_output.txt`

Reference output from a complete run of `sage/compute.sage`.

## Outline of the computation

The normalized cubical chain complex is filtered by support rank,

    F_1 ⊂ F_2 ⊂ F_3 ⊂ F_4.

The computation proceeds as follows. Here a non-degenerate cube means a cube that survives the quotient by degeneracies and connections. The term *based* means that the initial vertex is the identity permutation; the full chain complex retains all left translates.

1. Construct G_{5,4} and verify the cubical enumeration and normalization
   routines against direct reference implementations.

2. Compute the exact-support blocks of ranks one and two. The rank-one
   relative H_2 groups vanish, while each rank-two local block has
   one-dimensional H_2.

3. Compute H_1(F_1) and the connecting map delta_2, obtaining

       dim H_2(F_2) = 479.

4. Verify that every rank-three relative block has surjective d_3, so

       H_2(F_3,F_2) = 0.

5. Compute the connecting map delta_3 and verify

       rank(delta_3) = 479.

   Exactness therefore gives H_2(F_3) = 0.

6. Verify the corresponding relative surjectivity for all rank-four blocks.
   Hence H_2(F_4) = 0.

7. Exhaustive enumeration shows that every non-degenerate 2-cube has support rank
   at most four. Therefore the degree-two part of the full normalized cubical
   complex is already contained in F_4, and consequently

       DH_2(G_{5,4}; F_2) = 0.

## Verification

The implementation checks the following properties:
These include:

- comparison of optimized cubical routines with direct reference routines;
- exhaustive support checks for based non-degenerate 2-cubes;
- exhaustive local computations in support ranks one and two;
- SageMath rank cross-checks where the relevant matrices are small enough;
- reconstruction and replay of relative boundaries in support ranks three and four; and
- replay of independent images used to verify the rank of delta_3.

The reference output is provided only for comparison with a completed run; it
is not read by the computation.

## Relation to Appendix A

- A.1: cube enumeration, normalization, and support-rank filtration (`graph.sage`, `cubical.sage`).
- A.2: local relative boundary ranks (`filtration.sage`).
- A.3: connecting homomorphisms and the auxiliary quotient `Q` (`filtration.sage`). `Q` is not itself a homology group; cycle and kernel conditions are checked before interpreting the resulting vectors as homology classes.
- A.4: support-rank bound and verification (`compute.sage`, `results/compute_output.txt`).

The code retains internal identifiers such as `normalized_based_2` for backward compatibility. In the manuscript, these correspond to *based non-degenerate 2-cubes*. No mathematical algorithm or stored reference output is changed in this editorial revision.

## Expected conclusion

A successful complete run ends with:

```text
FINAL VERIFIED COMPUTATIONAL CONCLUSION: DH_2(G_{5,4}; F_2) = 0.
```

## License

The code in this repository is distributed under the MIT License. See
`LICENSE`.
