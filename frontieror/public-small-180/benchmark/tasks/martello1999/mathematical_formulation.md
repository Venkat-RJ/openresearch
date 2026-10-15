# Original Formulation (Eqs. 1a–1c)

## Sets and Indices

- $J = \{1, 2, \ldots, n\}$: set of items, indexed by $j \in J$.

## Parameters

- $n$: number of items (positive integer).

- $p_j$: profit of item $j$ (positive integer), for all $j \in J$.

- $w_j$: weight of item $j$ (positive integer), for all $j \in J$.

- $c$: knapsack capacity (positive integer).

Standard assumptions: $w_j \leq c$ for all $j$, and $\sum_{j=1}^{n} w_j > c$.

## Decision Variables

- $x_j \in \{0,1\}$: 1 if item $j$ is packed, 0 otherwise.

## Objective and Constraints

$$\begin{align}
\max \quad & z = \sum_{j=1}^{n} p_j\, x_j \tag{1a} \\
\text{s.t.}\quad & \sum_{j=1}^{n} w_j\, x_j \leq c \tag{1b} \\
& x_j \in \{0,1\}, \quad j \in \{1, \ldots, n\} \tag{1c}
\end{align}$$
