# Original Formulation: Maximum Clique Problem (MCP)

*Source: “Why Is Maximum Clique Often Easy in Practice?”, Jose L. Walteros, Austin Buchanan, 2020.*

## Sets and Parameters

$$\begin{align*}
G = (V, E) \quad &\text{simple undirected input graph}
  &&\% \text{RECONSTRUCTED --- paper's notation, Sec.\ 2} \\
V = \{1, \dots, n\}, \; n = |V| \quad &\text{set of vertices}
  &&\% \text{RECONSTRUCTED --- not in paper} \\
E \subseteq \tbinom{V}{2}, \; m = |E| \quad &\text{set of edges}
  &&\% \text{RECONSTRUCTED --- not in paper} \\
\overline{E} = \tbinom{V}{2} \setminus E \quad &\text{set of non-edges (edges of complement } \overline{G})
  &&\% \text{RECONSTRUCTED --- not in paper}
\end{align*}$$

## Decision Variables

$$\begin{align*}
x_v \in \{0,1\}, \quad v \in V \qquad
  &\text{$1$ if vertex $v$ is in the clique, $0$ otherwise.}
  &&\% \text{RECONSTRUCTED --- not in paper}
\end{align*}$$

## Objective

$$\begin{align}
\max \quad \sum_{v \in V} x_v
  \tag{R1} % RECONSTRUCTED --- maximize clique size omega(G)
\end{align}$$

## Constraints

$$\begin{align}
x_i + x_j &\le 1, &&\forall\, \{i,j\} \in \overline{E},
  \tag{R2} \\ % RECONSTRUCTED --- non-adjacent vertices cannot both be chosen
x_v &\in \{0,1\}, &&\forall\, v \in V.
  \tag{R3} % RECONSTRUCTED --- not in paper
\end{align}$$
