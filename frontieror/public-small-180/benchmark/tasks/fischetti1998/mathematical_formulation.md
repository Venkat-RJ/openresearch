# Original Formulation: Orienteering Problem (OP)

*Source: Solving the Orienteering Problem through Branch-and-Cut, Matteo Fischetti, Juan José Salazar González, Paolo Toth, 1998.*

## Sets and Indices

- $G=(V,E)$: a complete undirected graph.

- $V$: set of $n:=|V|$ nodes; node $1$ represents the depot.

- $E$: set of all undirected edges; an edge is written $[u,v]$ for $u,v\in V$, $u\neq v$.

- For $S\subset V$: $$E(S):=\{[u,v]\in E: u\in S,\, v\in S\},\qquad
        \delta(S):=\{[u,v]\in E: u\in S,\, v\notin S\}.$$

- For $v\in V$ we write $\delta(v)$ instead of $\delta(\{v\})$.

- For $T\subseteq E$: $V(T):=\{v\in V: T\cap\delta(v)\neq\varnothing\}$ (nodes spanned by $T$).

## Parameters

- $p_v\geq 0$: nonnegative prize associated with each node $v\in V$, with $p_1=0$.

- $t_e\geq 0$: nonnegative travel time associated with each edge $e\in E$ (symmetric: $t_{ij}=t_{ji}$).

- $t_0$: maximum total travel time allowed for the vehicle.

- All values $p_v$, $t_e$, and $t_0$ are assumed to be integer.

*Notation:* for any $F\subseteq E$, $x(F):=\sum_{e\in F}x_e$; for any $Q\subseteq V$, $y(Q):=\sum_{v\in Q}y_v$.

## Decision Variables

$$\begin{align*}
  x_e &:= \begin{cases} 1 & \text{if edge } e \text{ is used},\\ 0 & \text{otherwise},\end{cases} && e\in E;\\[4pt]
  y_v &:= \begin{cases} 1 & \text{if vertex } v \text{ is visited},\\ 0 & \text{otherwise},\end{cases} && v\in V.
\end{align*}$$

## Objective

$$\begin{equation}
  v(\mathrm{OP}) := \max \ \sum_{v\in V} p_v\, y_v \tag{1}
\end{equation}$$

## Constraints

$$\begin{align}
  & \sum_{e\in E} t_e\, x_e \leq t_0, \tag{2}\\
  & x(\delta(v)) = 2\, y_v && \text{for } v\in V, \tag{3}\\
  & x(\delta(S)) \geq 2\, y_v && \text{for } S\subset V,\ 1\in S,\ v\in V\setminus S, \tag{4}\\
  & y_1 = 1, \tag{5}\\
  & 0 \leq x_e \leq 1 && \text{for } e\in E, \tag{6}\\
  & 0 \leq y_v \leq 1 && \text{for } v\in V\setminus\{1\}, \tag{7}\\
  & x_e \text{ integer} && \text{for } e\in E, \tag{8}\\
  & y_v \text{ integer} && \text{for } v\in V\setminus\{1\}. \tag{9}
\end{align}$$

Constraint (2) imposes the bound $t_0$ on the total travel time. The degree equations (3) stipulate that a feasible solution must go exactly once through each visited node. The Generalized Subtour Elimination Constraints (GSECs) (4) force each visited node $v\in V\setminus\{1\}$ to be reachable from node 1 by means of two edge-disjoint paths. Constraint (5) imposes that node 1 must be visited, and (6)–(9) require that all variables are $0$–$1$ valued.
