# Original Formulation: Independent Set (IS)

## Sets and Indices

- $G = (V, E)$: undirected graph with vertex set $V$ ($|V| = n$) and edge set $E$.

- $\mathcal{S}$: the set of all maximal independent sets of $G$. An independent set is a subset of $V$ containing no two endpoints of any edge; a maximal independent set is one not strictly contained in any other independent set.

- $i \in V$: vertex index; $s \in \mathcal{S}$: maximal-independent-set index.

- Membership: $i \in s$ means vertex $i$ belongs to the independent set $s$.

## Decision Variables

- $x_s \in \{0, 1\}$ for each $s \in \mathcal{S}$: 1 if independent set $s$ is selected (assigned a unique color), 0 otherwise.

## Objective and Constraints

$$\begin{align}
\min \quad & \sum_{s \in \mathcal{S}} x_s
  \tag{1} \\
\text{s.t.}\quad & \sum_{s \in \mathcal{S}\, :\, i \in s} x_s \geq 1
  && \forall i \in V, \tag{2} \\
& x_s \in \{0, 1\}
  && \forall s \in \mathcal{S}. \tag{3}
\end{align}$$

The objective counts the number of selected maximal independent sets, which equals the number of colors used. Constraint (2) requires that each vertex be covered by at least one selected independent set.
