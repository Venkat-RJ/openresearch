# Original Formulation: Maximum Clique Problem (MCP)

San Segundo, Furini, Alvarez, Pardalos. “CliSAT: A new exact algorithm for hard maximum clique problems,” *EJOR*, 2023. Section 3.6, equations (9a)–(9c).

## Sets and Indices

- $G = (V,E)$: a simple undirected graph with vertex set $V$ and edge set $E$.

- $\bar{E}(G) = \bigl\{\{u,v\} : u,v\in V,\ u\neq v,\ \{u,v\}\notin E\bigr\}$: edge set of the complement graph $\bar G$ (pairs of non-adjacent vertices).

## Decision Variables

$$x_u \in \{0,1\}, \quad \forall u \in V,
  \qquad x_u = 1 \iff u \text{ belongs to the selected clique.}$$

## Natural ILP Formulation (9a)–(9c)

$$\begin{align}
  \omega(G) \ =\ \max\ & \sum_{u \in V} x_u \tag{9a}\\
  \text{s.t.}\ & x_u + x_v \le 1,
    && \forall\, \{u,v\} \in \bar{E}(G), \tag{9b}\\
  & x_u \in \{0,1\},
    && \forall\, u \in V. \tag{9c}
\end{align}$$

## Optional Strengthening (eq. 10)

The LP relaxation of (9a)–(9c) is weak (upper bound $\ge |V|/2$). The paper therefore replaces (9b) by independent-set inequalities: given a collection $\mathscr{C}$ of independent sets of $G$ covering all pairs $\{u,v\}\in\bar{E}(G)$, $$\begin{align}
  \sum_{u \in I} x_u \le 1, \quad \forall\, I \in \mathscr{C}. \tag{10}
\end{align}$$ The ILP used as the baseline in the paper’s computational experiments is (9a) + (10) + (9c); the original formulation as stated in the paper, however, is (9a)–(9c).
