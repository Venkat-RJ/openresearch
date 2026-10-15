# Original Formulation: Symmetric Generalized Traveling Salesman Problem (GTSP)

*Source: A Branch-and-Cut Algorithm for the Symmetric Generalized Traveling Salesman Problem, Matteo Fischetti, Juan José Salazar González, Paolo Toth, 1997.*

## Sets and Parameters

- $G=(N,E)$ — a complete (loop-free) undirected graph.

- $N:=\{1,\ldots,n\}$ — node set.

- $E:=\{\,[i,j] : i,j\in N,\; i\neq j\,\}$ — edge set.

- $C_1,C_2,\ldots,C_m$ — a proper partition of $N$ into node subsets (*clusters*); $m$ is the number of clusters.

- $c_e$ — cost associated with each edge $e\in E$ (integer in the computational study; assumed to satisfy the triangle inequality).

Derived notation (main notation used in the sequel). For each $S\subseteq N$: $$\begin{align*}
E(S) &:= \{\,[i,j]\in E : i\in S,\; j\in S\,\}, \\
\delta(S) &:= \{\,[i,j]\in E : i\in S,\; j\notin S\,\}, \\
\mu(S) &:= |\{\,h : C_h\subseteq S\,\}|, \\
\eta(S) &:= |\{\,h : C_h\cap S\neq\varnothing\,\}|.
\end{align*}$$ For $v\in N$ we write $\delta(v)$ instead of $\delta(\{v\})$, and denote by $C_{h(v)}$ the cluster containing $v$. We also define $W:=\{\,v\in N : |C_{h(v)}|=1\,\}$.

## Decision Variables

- $x_e\in\{0,1\}$ for each $e\in E$: $x_e=1$ if edge $e$ is chosen in the optimal solution, $0$ otherwise.

- $y_v\in\{0,1\}$ for each $v\in N$: $y_v=1$ if node $v$ is visited, $0$ otherwise.

## Objective

$$\begin{align}
v(\text{GTSP}):=\min \quad & \sum_{e\in E} c_e\, x_e \tag{1.1}
\end{align}$$

## Constraints

$$\begin{align}
\text{subject to}\quad
& \sum_{e\in\delta(v)} x_e = 2 y_v && \text{for } v\in N, \tag{1.2}\\[4pt]
& \sum_{v\in C_h} y_v \geq 1 && \text{for } h=1,\ldots,m, \tag{1.3}\\[4pt]
& \sum_{e\in\delta(S)} x_e \geq 2(y_i+y_j-1) && \text{for } S\subseteq N,\; 2\leq|S|\leq n-2, \notag\\
& && \quad i\in S,\; j\in N\setminus S, \tag{1.4}\\[4pt]
& x_e\in\{0,1\} && \text{for } e\in E, \tag{1.5}\\[4pt]
& y_v\in\{0,1\} && \text{for } v\in N. \tag{1.6}
\end{align}$$
