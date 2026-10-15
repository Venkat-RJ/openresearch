# Original Formulation: Directed Profitable Rural Postman Problem with Incompatibility Constraints (DPRPP-IC), Formulation (A)

*Source: The Rural Postman Problem with Incompatibility Constraints, Colombi, Corberán, Mansini, Plana, and Sanchis, 2017.*

## Sets and Parameters

- $G(V, A)$: strongly-connected directed graph with node set $V = \{0, 1, \dots, n\}$ (node $0$ is the depot) and arc set $A$.

- $R \subseteq A$: subset of profitable arcs (arcs that require service and yield positive profit).

- $V_I \subset V$: set of nodes $i \in V$ with at least one arc $(i,j) \in R$.

- $\bar{G}(\bar{V},\, E_1 \cup E_2)$: incompatibility graph with $\bar{V} \subseteq V_I$; edges $E_1$ are strong incompatibilities and $E_2$ are weak (removable via penalty) incompatibilities.

- $\delta^{+}(S)$, $\delta^{-}(S)$: arc cutsets leaving / entering $S \subseteq V$. $R(S)$: profitable arcs with both endpoints in $S$.

- $c_{ij}$: traveling cost on arc $(i,j) \in A$.

- $p_{ij}$: positive profit on each profitable arc $(i,j) \in R$ (collected once per arc, the first time it is traversed).

- $\bar{c}_{ij}$: penalty cost to remove a weak incompatibility $\{i,j\} \in E_2$.

## Decision Variables

- $x_{ij} \geq 0$, integer, for $(i,j) \in A$: number of times arc $(i,j)$ is traversed.

- $y_{ij} \in \{0,1\}$ for $(i,j) \in R$: $=1$ iff profitable arc $(i,j)$ is served.

- $z_i \in \{0,1\}$ for $i \in \bar{V}$: $=1$ iff at least one profitable arc leaving $i$ is served.

- $u_{ij} \in \{0,1\}$ for $\{i,j\} \in E_2$: $=1$ iff the weak-incompatibility penalty between $i$ and $j$ is paid.

## Objective

$$\begin{equation}
w \;=\; \max \;\; \sum_{(i,j) \in R} p_{ij}\, y_{ij}
                \;-\; \sum_{(i,j) \in A} c_{ij}\, x_{ij}
                \;-\; \sum_{\{i,j\} \in E_2} \bar{c}_{ij}\, u_{ij}
\tag{A}
\end{equation}$$

## Constraints

$$\begin{align}
x_{ij} & \;\geq\; y_{ij}, & (i,j) \in R \tag{1} \\[2pt]
\sum_{(j,i) \in \delta^{+}(j)} x_{ji}
  \;-\; \sum_{(i,j) \in \delta^{-}(j)} x_{ij} & \;=\; 0, & j \in V \tag{2} \\[2pt]
\sum_{(i,j) \in \delta^{+}(S)} x_{ij} & \;\geq\; y_{ks}, & S \subseteq V \setminus \{0\},\; (k,s) \in R(S) \tag{3} \\[2pt]
y_{ij} & \;\leq\; z_{i}, & i \in \bar{V},\; (i,j) \in R \tag{4} \\[2pt]
z_i + z_j & \;\leq\; 1, & \{i,j\} \in E_1 \tag{5} \\[2pt]
z_i + z_j - u_{ij} & \;\leq\; 1, & \{i,j\} \in E_2 \tag{6} \\[2pt]
x_{ij} & \;\geq\; 0, \text{ integer}, & (i,j) \in A \tag{7} \\[2pt]
y_{ij} & \;\in\; \{0,1\}, & (i,j) \in R \tag{8} \\[2pt]
z_{i} & \;\in\; \{0,1\}, & i \in \bar{V} \tag{9} \\[2pt]
u_{ij} & \;\in\; \{0,1\}, & \{i,j\} \in E_2 \tag{10}
\end{align}$$

By Proposition 1 of the paper, constraints (9)–(10) may be relaxed to $z_i \in [0,1]$ and $u_{ij} \in [0,1]$ without loss of optimality.
