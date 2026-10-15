# Original Formulation: Edge Interdiction Clique Problem (EICP)

*Source: A branch-and-cut algorithm for the Edge Interdiction Clique Problem, Fabio Furini, Ivana Ljubić, Pablo San Segundo, Yanlu Zhao, 2021 (European Journal of Operational Research 294, 54–69).*

## Sets and Parameters

- $G=(V,E)$: a simple undirected graph with $n=|V|$ vertices and $m=|E|$ edges.

- $\bar{E}$: the complement edge set, i.e. the edges of $\bar{G}=(V,\bar{E})$ (pairs of vertices *not* adjacent in $G$).

- $k \in \mathbb{N}$: the interdiction budget (maximum number of edges that may be removed). It is assumed w.l.o.g. that $k < |E|$.

- $uv$: an (undirected) edge between vertices $u$ and $v$.

## Decision Variables

- $w_{uv} \in \{0,1\}$, $uv \in E$ (leader): $1$ if edge $uv$ is interdicted (removed), $0$ otherwise.

- $x_v \in \{0,1\}$, $v \in V$ (follower): $1$ if vertex $v$ belongs to a maximum clique of the interdicted graph $G_I=(V,E\setminus\{uv:w_{uv}=1\})$.

- $\vartheta$: auxiliary variable representing the size of the maximum clique in the interdicted graph (the value of the inner maximization problem).

## Objective

$$\begin{align}
\min \quad & \vartheta \tag{1a}
\end{align}$$

## Constraints

The EICP is stated as the following bilevel optimization problem. The outer (leader) problem chooses which edges to interdict; the inner (follower) problem, embedded in constraint (1d), finds a maximum clique in the interdicted graph. $$\begin{align}
\text{subject to (s.t.)} \quad & \sum_{uv \in E} w_{uv} \le k \tag{1b} \\
& w_{uv} \in \{0,1\} && uv \in E \tag{1c} \\
\text{where} \quad & \vartheta = \max \ \sum_{u \in V} x_u \tag{1d} \\
\text{s.t.} \quad & x_u + x_v \le 2 - w_{uv} && uv \in E \tag{1e} \\
& x_u + x_v \le 1 && uv \in \bar{E} \tag{1f} \\
& x_v \in \{0,1\} && v \in V. \tag{1g}
\end{align}$$
