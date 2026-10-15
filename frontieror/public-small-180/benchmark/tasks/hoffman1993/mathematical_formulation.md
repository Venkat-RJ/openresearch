# Sets and Parameters

- $m$: number of flight legs; index $i \in \{1,\ldots,m\}$.

- $n$: number of feasible rotations; index $j \in \{1,\ldots,n\}$.

- $d$: number of base (side) constraints.

- $A \in \{0,1\}^{m \times n}$: leg-rotation matrix; $a_{ij}=1$ iff rotation $j$ covers flight leg $i$.

- $c_j \in \mathbb{R}$: cost of rotation $j$ (incremental costs).

- $D \in \mathbb{Q}_{+}^{d \times n}$: base-constraint coefficient matrix.

- $d_1, d_2 \in \mathbb{Q}^{d}$: lower/upper bounds for base constraints, with $d_1 \leq d_2$.

- $e_m = (1,1,\ldots,1)^{\top} \in \mathbb{R}^{m}$: vector of $m$ ones.

# Decision Variables

- $x_j \in \{0,1\}$, $j = 1,\ldots,n$: $x_j = 1$ iff rotation $j$ is selected.

# Formulation (SPB)

$$\begin{align}
\min \quad & \sum_{j=1}^{n} c_j\, x_j \\
\text{s.t.}\quad & A x = e_m, \tag{SPB-a} \\
& d_1 \leq D x \leq d_2, \tag{SPB-b} \\
& x \in \{0,1\}^{n}. \tag{SPB-c}
\end{align}$$

(SPB-a) is the partitioning condition: each flight leg is covered by exactly one rotation. (SPB-b) are the base (work-rule) side constraints. (SPB-c) is integrality.
