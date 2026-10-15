# Original Formulation: Orienteering Problem (OP)

*Source: A revisited branch-and-cut algorithm for large-scale orienteering problems, Gorka Kobeaga, Jairo Rojas-Delgado, María Merino, Jose A. Lozano, European Journal of Operational Research 313 (2024) 44–68.*

## Sets and Parameters

The OP is defined by the 5-tuple $(G, d, s, o, d_0)$, where:

- $G = K_n = (V, E)$ is a complete graph with vertex set $V$ and edge set $E$.

- $d = (d_e)$, where $d_e > 0$ is the distance value (time or weight) associated with each edge $e \in E$.

- $s = (s_v)$, where $s_v > 0$ is the score (profit) of vertex $v \in V$.

- $o \in V$ is the vertex selected as the depot.

- $d_0 > 0$ is the value that limits the cycle length.

The following edge/vertex sets are used: $$\begin{align}
(Q : W) &:= \{[u, v] \in E : u \in Q,\, v \in W\}, & Q, W &\subseteq V \tag{1a} \\
\delta(Q) &:= (Q : V - Q), & Q &\subseteq V \tag{1b} \\
E(Q) &:= (Q : Q), & Q &\subseteq V \tag{1c} \\
V(T) &:= \{v \in V : T \cap (v : V) \neq \emptyset\}, & T &\subseteq E \tag{1d}
\end{align}$$ where $(Q:W)$ is the set of edges connecting $Q$ and $W$, $\delta(Q)$ is the coboundary (star-set) of $Q$, $E(Q)$ is the set of edges between the vertices of $Q$, and $V(T)$ is the set of vertices incident with an edge set $T$. For $S \subseteq V$ and $T \subseteq E$, we write $y(S) = \sum_{v \in S} y_v$ and $x(T) = \sum_{e \in T} x_e$.

## Decision Variables

Two types of variables, $y = (y_v) \in \mathbb{R}^V$ and $x = (x_e) \in \mathbb{R}^E$, associated with the nodes and edges of $G$ respectively: $$\begin{align*}
y_v &= \begin{cases} 1 & \text{if node } v \text{ is visited} \\ 0 & \text{otherwise} \end{cases} & v &\in V \\[4pt]
x_e &= \begin{cases} 1 & \text{if edge } e \text{ is traversed} \\ 0 & \text{otherwise} \end{cases} & e &\in E
\end{align*}$$

## Objective

Maximize the total collected profit (sum of the scores of the visited vertices): $$\begin{align}
\max \quad & \sum_{v \in V} s_v y_v \tag{2a}
\end{align}$$

## Constraints

$$\begin{align}
\text{s.t.} \quad
& \sum_{e \in E} d_e x_e \leq d_0 \tag{2b} \\
& x(\delta(v)) - 2 y_v = 0, & v &\in V \tag{2c} \\
& x(\delta(H)) - 2 y_l - 2 y_r \geq -2, & l &\in H \subseteq V,\ r \in V - H \tag{2d} \\
& \qquad 3 \leq |H| \leq |V| - 3 \notag \\
& y_v - x_e \geq 0, & v &\in V,\ e \in \delta(v) \tag{2e} \\
& 0 \leq y_v \leq 1, & v &\in V \tag{2f} \\
& 0 \leq x_e \leq 1, & e &\in E \tag{2g} \\
& y_o = 1 \tag{2h} \\
& x_e \in \mathbb{Z}, & e &\in E \tag{2i}
\end{align}$$
