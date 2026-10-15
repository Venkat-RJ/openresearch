# Sets and Indices

- $N = \{1, \ldots, n\}$: set of items, indexed by $i, j$.

# Parameters

- $p_{ij} \in \mathbb{Z}_+$: symmetric profit matrix ($p_{ij} = p_{ji}$). Diagonal entries $p_{jj}$ are linear profits; off-diagonal entries represent the pairwise profit of selecting both items $i$ and $j$.

- $w_j \in \mathbb{Z}_+$: weight of item $j$, for $j = 1,\ldots,n$.

- $c \in \mathbb{Z}_+$: knapsack capacity.

# Decision Variables

- $x_j \in \{0,1\}$: equals $1$ if item $j$ is selected and $0$ otherwise, for $j = 1,\ldots,n$.

# Formulation (1)

$$\begin{align}
  \max \quad & \sum_{i=1}^{n} \sum_{j=1}^{i} p_{ij}\, x_i\, x_j \tag{1} \\
  \text{s.t.}\quad & \sum_{j=1}^{n} w_j\, x_j \le c \notag \\
  & x_j \in \{0,1\}, \qquad j = 1, \ldots, n \notag
\end{align}$$

The objective is quadratic in the binary variables; each unordered pair (including the diagonal) is counted exactly once by summing over $j \le i$.
