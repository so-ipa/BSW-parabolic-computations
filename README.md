# Computation of DH_2(G_{5,4}; F_2)

This repository contains the SageMath code used to reproduce the computation

    DH_2(G_{5,4}; F_2) = 0.

The computation uses a filtration by Coxeter support and performs the required
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
graph cubes, cubical faces, normalization, Coxeter support, and the enumeration
of based cubes. It also contains direct reference implementations used to check
the optimized routines.

### `sage/gf2.py`

Implements low-level bitset linear algebra over F_2, including streaming
Gaussian elimination and kernel-relation tracking.

### `sage/filtration.sage`

Implements the support-filtration computation: local exact-support blocks, the
first filtration stage, the connecting maps, and the relative surjectivity
calculations used in the proof.

### `sage/compute.sage`

Main reproducibility script. It constructs G_{5,4}, verifies the cubical
infrastructure, runs the complete support-filtration computation, and checks
the deductions leading to the final vanishing result.

### `results/compute_output.txt`

Reference output from a complete run of `sage/compute.sage`.

## Outline of the computation

The normalized cubical chain complex is filtered by Coxeter support,

    F_1 ⊂ F_2 ⊂ F_3 ⊂ F_4.

The computation proceeds as follows.

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

7. Exhaustive enumeration shows that every normalized 2-cube has support rank
   at most four. Therefore the degree-two part of the full normalized cubical
   complex is already contained in F_4, and consequently

       DH_2(G_{5,4}; F_2) = 0.

## Verification

The implementation includes internal checks intended to make the computation
reproducible.
These include:

- comparison of optimized cubical routines with direct reference routines;
- exhaustive support checks for normalized based 2-cubes;
- exhaustive local computations in support ranks one and two;
- SageMath rank cross-checks where the relevant matrices are small enough;
- replay of explicit boundary certificates in the higher relative
  computations; and
- replay of independent images used to certify the rank of delta_3.

The reference output is provided only for comparison with a completed run; it
is not read by the computation.

## Expected conclusion

A successful complete run ends with:

```text
FINAL VERIFIED COMPUTATIONAL CONCLUSION: DH_2(G_{5,4}; F_2) = 0.
```

## License

The code in this repository is distributed under the MIT License. See
`LICENSE`.
