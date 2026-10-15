# Original Formulation: Min-Power Symmetric Connectivity (MinPSC)

*Source: Parameterized Algorithms for Power-Efficiently Connecting Wireless Sensor Networks: Theory and Experiments, Matthias Bentert, René van Bevern, André Nichterlein, Rolf Niedermeier, Pavel V. Smirnov, 2020 (arXiv:1706.03177v3).*

## Sets and Parameters

- $G = (V, E)$: a connected, undirected, finite, simple graph.

- $V$: set of vertices (wireless sensors), $|V| = n$.

- $E \subseteq \big\{\,\{v, w\} \mid v \neq w \text{ and } v, w \in V\,\big\}$: set of undirected edges, $|E| = m$.

- $w : E \to \mathbb{N}$: edge-weight function; weights are natural numbers (the paper takes $\mathbb{N}$ to include $0$).

- Convention (Section 2.1): $\max \emptyset = -\infty$ and $\min \emptyset = \infty$.

## Decision Variables

- $F \subseteq E$: the set of edges selected for the spanning subgraph $T = (V, F)$. (The vertex set of $T$ is fixed to $V$; $T$ *spans* $G$.)

## Objective

Find a connected spanning subgraph $T = (V, F)$, $F \subseteq E$, of $G$ that minimizes $$\begin{equation*}
  \sum_{v \in V} \ \max_{\{u, v\} \in F} w(\{u, v\}).
  \tag{Problem 1.1}
\end{equation*}$$ That is, each vertex $v$ pays the most expensive edge incident to it in the solution, and the cost of $T$ is the sum of these payments. The minimum cost of a solution to an instance $I = (G, w)$ is denoted $\mathrm{Opt}(I)$.

## Constraints

$$\begin{align*}
  & T = (V, F), \quad F \subseteq E
    && \text{(spanning subgraph: vertex set is all of $V$)} \tag{C1} \\
  & T = (V, F) \text{ is connected}
    && \text{(every pair of vertices joined by a path in $F$)} \tag{C2}
\end{align*}$$
