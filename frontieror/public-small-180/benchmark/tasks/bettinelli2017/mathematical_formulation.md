# Original Formulation: Knapsack Problem with Conflict Graph (KPCG)

*Source: A Branch-and-Bound Algorithm for the Knapsack Problem with Conflict Graph, Andrea Bettinelli, Valentina Cacchiani, Enrico Malaguti, 2017 (INFORMS Journal on Computing, 29(3):457–473).*

## Sets and Indices

- $n$: number of items.

- $i, j$: item indices, $i, j = 1, \ldots, n$.

- $\mathcal{G} = (\mathcal{V}, \mathcal{E})$: undirected conflict graph; each vertex $i \in \mathcal{V}$ corresponds to one item, so $n = |\mathcal{V}|$.

- $\mathcal{E}$: set of edges of the conflict graph; an edge $(i,j) \in \mathcal{E}$ denotes that items $i$ and $j$ cannot be packed together.

## Parameters

- $c$: positive integer capacity of the knapsack.

- $p_i$: positive integer profit of item $i$,  $i = 1, \ldots, n$.

- $w_i$: positive integer weight of item $i$,  $i = 1, \ldots, n$.

*Assumptions (without loss of generality):* $\sum_{i=1,\ldots,n} w_i > c$,  $w_i \le c$ for all $i = 1, \ldots, n$, and items are sorted by nonincreasing profit-over-weight ratio $\frac{p_1}{w_1} \ge \frac{p_2}{w_2} \ge \cdots \ge \frac{p_n}{w_n}$.

## Decision Variables

- $x_i \in \{0,1\}$,  $i = 1, \ldots, n$:  $x_i = 1$ if item $i$ is selected (packed into the knapsack), $0$ otherwise.

## Objective

$$\begin{align}
\text{Maximize} \quad & \sum_{i=1,\ldots,n} p_i x_i \tag{1a}
\end{align}$$

## Constraints

$$\begin{align}
\text{s.t.} \quad & \sum_{i=1,\ldots,n} w_i x_i \le c, \tag{1b} \\
& x_i + x_j \le 1 && (i,j) \in \mathcal{E}, \tag{1c} \\
& x_i \in \{0,1\} && i = 1, \ldots, n. \tag{1d}
\end{align}$$

The objective (1a) maximizes the sum of the profits of the selected items. Constraint (1b) requires not exceeding the capacity of the knapsack. Constraints (1c) impose choosing at most one item for each conflicting pair (edge) of the conflict graph. Constraints (1d) require the variables to be binary.
