# Original Formulation: Maximum Clique Problem (MCP)

*Source: A new exact maximum clique algorithm for large and massive sparse graphs, Pablo San Segundo, Alvaro Lopez, Panos M. Pardalos, 2016.*

## Sets and Parameters

- $G=(V,E)$: a simple undirected graph (paper, Section 1).

- $V=\{v_1,v_2,\dots,v_n\}$: finite set of vertices, $|V|=n$.

- $E\subseteq V\times V$: set of edges, made up of pairs of distinct vertices.

- Two vertices are *adjacent* (neighbors) iff they are connected by an edge.

- $N(v)$ (or $N_G(v)$): the neighbor set of vertex $v$ in $G$, i.e. $\{u\in V:(u,v)\in E\}$.

- $\omega(G)$: the clique number, the size of the maximum clique (= optimal objective value).

## Decision Variables

- $x_i\in\{0,1\}$, $\forall i\in V$: $x_i=1$ if vertex $v_i$ is selected into the clique, $0$ otherwise.

## Objective

$$\begin{align}
\max \quad \omega \;=\; \sum_{i\in V} x_i \tag{OBJ}
\end{align}$$

## Constraints

$$\begin{align}
x_i + x_j &\;\le\; 1, & \forall\,(i,j)\notin E,\ i<j \tag{C1}\\[2pt]
x_i &\;\in\; \{0,1\}, & \forall\, i\in V \tag{C2}
\end{align}$$
