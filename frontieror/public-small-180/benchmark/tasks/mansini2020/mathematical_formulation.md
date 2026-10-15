# Original Formulation (Eqs. 1–4)

## Sets and Indices

- $\{G_1, G_2, \ldots, G_n\}$: a collection of $n$ disjoint groups (classes) of items, with $G_q \cap G_l = \emptyset$ for all $q \neq l$.

- Each group $G_i$ has $n_i = |G_i|$ items, denoted by ordered pairs $(i, j)$, $i = 1,\ldots,n$, $j = 1,\ldots,n_i$.

- $\tilde{N}$: set of all items, $|\tilde{N}| = \tilde{n} = \sum_{i=1}^{n} n_i$.

- $k = 1,\ldots,m$: index for the $m$ resources.

## Parameters

- $p_{ij} \geq 0$: profit of item $(i,j) \in \tilde{N}$.

- $w_{ij}^{k} \geq 0$: amount of resource $k$ required by item $(i,j)$.

- $c_k \geq 0$: capacity of resource $k$, for $k=1,\ldots,m$.

## Decision Variables

- $x_{ij} \in \{0,1\}$ for each $(i,j) \in \tilde{N}$: 1 if item $(i,j)$ is selected, 0 otherwise.

## Objective and Constraints

$$\begin{align}
\max \quad & z = \sum_{i=1}^{n} \sum_{j=1}^{n_i} p_{ij}\, x_{ij}
  \tag{1} \\
\text{s.t.}\quad & \sum_{i=1}^{n} \sum_{j=1}^{n_i} w_{ij}^{k}\, x_{ij} \leq c_k
  && k = 1,\ldots,m, \tag{2} \\
& \sum_{j=1}^{n_i} x_{ij} = 1
  && i = 1,\ldots,n, \tag{3} \\
& x_{ij} \in \{0,1\}
  && (i,j) \in \tilde{N}. \tag{4}
\end{align}$$
