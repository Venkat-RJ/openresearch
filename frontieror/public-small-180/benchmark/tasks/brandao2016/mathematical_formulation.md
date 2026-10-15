# Original Formulation: Bin Packing / Cutting Stock via General Arc-flow (AF)

*Source: Bin Packing and Related Problems: General Arc-flow Formulation with Graph Compression, Filipe Brandão and João Pedro Pedroso, 2013.*

## Sets and Indices

- $m$ : number of different item types (weights), indexed $i = 1, \ldots, m$.

- $V$ : set of vertices of the directed acyclic graph $G = (V, A)$; includes the source vertex $\textsc{s}$ and the target vertex $\textsc{t}$.

- $A$ : set of arcs. Each arc is a triple $(u, v, i)$ where $u, v \in V$ and $i$ identifies the item type contributing to the arc; arcs with $i = 0$ are the *loss* arcs (representing unused capacity).

- $J \subseteq \{1, \ldots, m\}$ : subset of items whose demands must be satisfied *exactly* (for efficiency the authors set $J = \{ i = 1, \ldots, m \mid b_i = 1 \}$).

## Parameters

- $b_i$ : demand of items of type $i$, for $i = 1, \ldots, m$.

- $w_i$ : weight (size) of item type $i$.

- $W$ : bin capacity.

- $\textsc{s}, \textsc{t}$ : source and target vertices of $G$.

*The graph $G = (V, A)$ is constructed beforehand so that every path from $\textsc{s}$ to $\textsc{t}$ corresponds to a valid packing pattern for a single bin; $V$ and $A$ are therefore inputs to the MILP below.*

## Decision Variables

- $f_{uvi}$ : amount of flow along arc $(u, v, i) \in A$ (non-negative integer).

- $z$ : number of bins required (total flow from $\textsc{s}$ to $\textsc{t}$).

## Objective

$$\begin{align}
\text{minimize} \quad & z \tag{17}
\end{align}$$

## Constraints

$$\begin{align}
\sum_{(u,v,i) \in A:\, v = k} f_{uvi}
  \;-\; \sum_{(v,r,i) \in A:\, v = k} f_{vri}
  &=
  \begin{cases}
    -z & \text{if } k = \textsc{s}, \\
    \phantom{-}z & \text{if } k = \textsc{t}, \\
    \phantom{-}0 & \text{for } k \in V \setminus \{\textsc{s}, \textsc{t}\},
  \end{cases} \tag{18} \\[4pt]
\sum_{(u,v,j) \in A:\, j = i} f_{uvj} &\geq b_i,
  && i \in \{1, \ldots, m\} \setminus J, \tag{19} \\[4pt]
\sum_{(u,v,j) \in A:\, j = i} f_{uvj} &= b_i,
  && i \in J, \tag{20} \\[4pt]
f_{uvi} &\leq b_i,
  && \forall (u,v,i) \in A, \text{ if } i \neq 0, \tag{21} \\[4pt]
f_{uvi} &\geq 0, \text{ integer},
  && \forall (u,v,i) \in A. \tag{22}
\end{align}$$
