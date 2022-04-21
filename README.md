A near-term quantum algorithm for solving linear systems of equations based on the Woodbury identity
===============================

Description
-----------

This code utilizes the [Woodbury identity](https://en.wikipedia.org/wiki/Woodbury_matrix_identity) to solve linear systems of equations on a quantum computer. At present, it currently supports a limited subset of the Woodbury identity. Using the notation from the Wikipedia page, the code requires that A=I, C=I, and U and V both are rank 1. The code could be tweaked to solve more complex versions of the Woodbury identity. The method was able to estimate a quantity of interest based on the solution to a system of ~16 million equations with only ~2% error using IBM's Auckland quantum computer.

The algorithm is near-term because it only relies on the [Hadamard test](https://en.wikipedia.org/wiki/Hadamard_test_(quantum_computation)), which is a simple quantum subroutine. Other quantum algorithms for solving linear systems that are guaranteed to succeed with high probably, like [HHL](https://arxiv.org/abs/0811.3171), use complex quantum subroutines. These complex subroutines are only suitable for fault-tolerant quantum computers, which do not yet exist. Variational quantum algorithms for solving linear systems of equations, like [VQLS](https://arxiv.org/abs/1909.05820), are more near-term than HHL-like algorithms, but require an optimization loop to find the solution, which is expensive and prone to barren plateaus and local optima. The approach used here provides the best of both worlds, but is limited to the Woodbury niche.

License
-------

Origami is provided under a BSD-ish license with a "modifications must be indicated" clause.  See LICENSE.md file for the full text.

This package is part of the Hybrid Quantum-Classical Computing suite, known internally as LA-CC-16-032.

Author
------

Daniel O'Malley, <omalled@lanl.gov>

