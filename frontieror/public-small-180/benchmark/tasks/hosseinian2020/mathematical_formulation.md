# Sets and Parameters

- $G = (V, E, w^{E})$: undirected edge-weighted graph.

- $V = \{1,\ldots,n\}$: set of vertices.

- $E \subseteq \binom{V}{2}$: set of edges.

- $w_{ij} > 0$: weight of edge $\{i,j\} \in E$.

# Decision Variables

- $x_i \in \{0,1\}$ for all $i \in V$: $x_i = 1$ iff vertex $i$ is in the clique.

- $y_{ij} \in \{0,1\}$ for all $\{i,j\} \in E$: $y_{ij} = 1$ iff edge $\{i,j\}$ is in the clique.

# Formulation (Eq. (1) in the paper)

$$\begin{align}
W^{*} = \max \quad & \sum_{\{i,j\} \in E} w_{ij}\, y_{ij} \tag{1} \\
\text{s.t.}\quad & y_{ij} \leq x_i, && \forall\, \{i,j\} \in E, \\
& y_{ij} \leq x_j, && \forall\, \{i,j\} \in E, \\
& x_i + x_j \leq 1, && \forall\, \{i,j\} \notin E,\; i < j, \\
& x_i \in \{0,1\}, && \forall\, i \in V, \\
& y_{ij} \in \{0,1\}, && \forall\, \{i,j\} \in E.
\end{align}$$

The first two inequalities ensure an edge can only be selected if both its endpoints are selected. The non-adjacency constraints force the selected vertex set to induce a clique.
