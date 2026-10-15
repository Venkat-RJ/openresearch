# Sets and Indices

- $G = (V, E)$: a simple undirected graph with vertex set $V$ and edge set $E$.

- $\mathscr{I}$: the collection of all independent sets of $G$ (i.e., subsets $I \subseteq V$ such that no two vertices of $I$ are joined by an edge in $E$).

# Parameters

- $w: V \to \mathbb{R}_+$: a positive weight associated with each vertex $v \in V$.

# Decision Variables

- $x_v \in \{0,1\}$ for each $v \in V$: equals $1$ if vertex $v$ is selected in the clique, $0$ otherwise.

# Objective

Maximize the total weight of vertices selected in the clique: $$\begin{align}
  \omega(G, w) \;=\; \max \quad & \sum_{v \in V} w(v)\, x_v \tag{2}
\end{align}$$

# Constraints

Independent-set (clique-on-complement) constraints: for each independent set of $G$, at most one vertex may be chosen in the clique. $$\begin{align}
  \sum_{v \in I} x_v &\le 1, && \forall\, I \in \mathscr{I} \tag{3} \\
  x_v &\in \{0,1\}, && \forall\, v \in V \tag{4}
\end{align}$$
