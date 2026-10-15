# Sets and Parameters

- $I$: set of customers, indexed by $i$.

- $J$: set of potential plant (facility) locations, indexed by $j$.

- $c_{ij} \geq 0$: cost of supplying all of customer $i$’s demand from $j$.

- $f_j \geq 0$: fixed cost of opening facility $j$.

- $d_i \geq 0$: demand of customer $i$.

- $s_j > 0$: capacity of facility $j$ when open.

- $d^{I} = \sum_{i \in I} d_i$: total demand. Assumed: $\sum_{j \in J} s_j > d^{I}$.

# Decision Variables

- $y_j \in \{0,1\}$ for $j \in J$: $=1$ iff facility $j$ is open.

- $x_{ij} \in [0,1]$ for $i \in I,\ j \in J$: fraction of customer $i$’s demand met from $j$.

# Formulation

$$\begin{align}
Z = \min \quad & \sum_{i \in I}\sum_{j \in J} c_{ij}\, x_{ij} + \sum_{j \in J} f_j\, y_j \\
\text{s.t.}\quad & \sum_{j \in J} x_{ij} = 1,
   && \forall i \in I, \tag{D}\\
& \sum_{i \in I} d_i\, x_{ij} \leq s_j\, y_j,
   && \forall j \in J, \tag{C}\\
& 0 \leq x_{ij} \leq 1,
   && \forall i \in I,\ j \in J, \tag{N}\\
& y_j \in \{0,1\},
   && \forall j \in J. \tag{I}
\end{align}$$

\(D\) enforces full demand satisfaction; (C) enforces capacity and links allocation to opening; (N) are bounds on the continuous allocation variables; (I) is integrality of the location variables.
