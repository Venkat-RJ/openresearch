# Original Formulation: Capacitated Arc-Routing Problem (CARP) – Two-Index Formulation

*Source: “Cut-First Branch-and-Price-Second for the Capacitated Arc-Routing Problem,” Bode and Irnich, Operations Research 60(5):1167–1182, 2012.*

## Sets and Parameters

- $G = (V, E)$: undirected graph with node set $V$ and edge set $E$.

- $E_R \subseteq E$: subset of required edges (those with positive demand).

- $K$: fleet of $|K|$ homogeneous vehicles, stationed at depot $d \in V$.

- $q_e \ge 0$: integer demand on edge $e \in E$; $Q$: vehicle capacity.

- $c^{\mathrm{serv}}_e$: service cost for required edge $e \in E_R$; $c_e$: deadheading cost for edge $e \in E$.

- Notation: for $S \subseteq V$, $\delta(S)$ is the set of edges with exactly one endpoint in $S$, $\delta_R(S) = \delta(S) \cap E_R$, $E_R(S) = E(S) \cap E_R$; $\delta(i) := \delta(\{i\})$; for variable $y$ and subset $F$, $y(F) = \sum_{e \in F} y_e$.

## Decision Variables

- $x^k_e \in \{0,1\}$ for $e \in E_R$, $k \in K$: $1$ if vehicle $k$ services edge $e$.

- $y^k_e \in \mathbb{Z}_+$ for $e \in E$, $k \in K$: number of times vehicle $k$ traverses edge $e$ without servicing it.

- $p^k_i \in \mathbb{Z}_+$ for $i \in V$, $k \in K$: auxiliary parity variable at node $i$ for vehicle $k$.

## Objective

$$\begin{equation}
\min \sum_{k \in K} {c^{\mathrm{serv}}}^{\!\top} x^k + \sum_{k \in K} c^{\top} y^k \tag{1}
\end{equation}$$

## Constraints

$$\begin{align}
\sum_{k \in K} x^k_e &= 1, & \forall\, e \in E_R \tag{2} \\
x^k(\delta_R(S)) + y^k(\delta(S)) &\ge 2\, x^k_f, & \forall\, S \subseteq V \setminus \{d\},\; f \in E_R(S),\; k \in K \tag{3} \\
x^k(\delta_R(i)) + y^k(\delta(i)) &= 2\, p^k_i, & \forall\, i \in V,\; k \in K \tag{4} \\
q^{\top} x^k &\le Q, & \forall\, k \in K \tag{5} \\
p^k \in \mathbb{Z}_+^{|V|},\; x^k &\in \{0,1\}^{|E_R|},\; y^k \in \mathbb{Z}_+^{|E|}, & \forall\, k \in K \tag{6}
\end{align}$$
