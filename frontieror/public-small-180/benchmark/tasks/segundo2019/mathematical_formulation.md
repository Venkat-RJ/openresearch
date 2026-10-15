# Original Formulation: Maximum Edge-Weighted Clique Problem (MEWCP)

*Source: A new branch-and-bound algorithm for the maximum edge-weighted clique problem, Pablo San Segundo, Stefano Coniglio, Fabio Furini, Ivana Ljubić, 2019.*

- $G=(V,E)$: simple undirected graph with $|V|$ vertices and $|E|$ edges.

- $C \subseteq V$: a clique (every pair of its vertices is joined by an edge of $E$).

- $E[C] := \{\, e=\{u,v\}\in E : u,v \in C \,\}$: edges with both endpoints in $C$.

- $N(u) := \{\, v\in V : \{u,v\}\in E \,\}$: neighborhood of vertex $u$.

- $\overline{G}=(V,\overline{E})$: complement graph, where $\overline{E} := \{\, e=\{u,v\}, u,v\in V : e\notin E \,\}$ is the set of *non-edges* of $G$.

- $\mathscr{I}$: the set of all (maximal) independent sets of $G$ (i.e. cliques of $\overline{G}$).

- $\delta(u) \subseteq E$: subset of edges incident to vertex $u$.

- $\omega(G)$: clique number of $G$ (number of vertices of a largest clique).

## Parameters

- $c_e \in \mathbb{R}^{+}$: strictly positive weight of each edge $e\in E$ (weight function $c:E\to\mathbb{R}^{+}$).

## Decision Variables

- $x_u \in \{0,1\}$, $u\in V$: $x_u=1$ iff vertex $u$ belongs to the chosen clique.

- $y_e \in \{0,1\}$, $e=\{u,v\}\in E$: $y_e=1$ iff both endpoints $u$ and $v$ of edge $e$ belong to the chosen clique.

## Objective

The MEWCP seeks a clique maximizing the total weight of the edges whose endpoints are both in the clique: $$\begin{align}
\max \quad & \sum_{e \in E} c_e\, y_e \tag{4}
\end{align}$$

## Constraints

$$\begin{align}
\text{s.t.}\quad
& \sum_{v \in I} x_v \;\le\; 1, && I \in \mathscr{I} \tag{5}\\
& y_e \;\le\; x_u, && e=\{u,v\}\in E \tag{6}\\
& y_e \;\le\; x_v, && e=\{u,v\}\in E \tag{7}\\
& x_u \in \{0,1\}, && u \in V \tag{8}\\
& y_e \in \{0,1\}, && e \in E \tag{9}
\end{align}$$
