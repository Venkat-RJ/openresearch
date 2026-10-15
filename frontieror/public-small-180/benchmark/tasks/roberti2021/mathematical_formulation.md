# Original Formulation: Traveling Salesman Problem with Drone (TSP-D)

*Source: Exact Methods for the Traveling Salesman Problem with Drone, Roberto Roberti and Mario Ruthmair, Transportation Science, 55(2):315–335, 2021.*

## Sets and Indices

- $V = \{0, 0'\} \cup N$ : vertex set of the complete directed graph $\mathcal{G} = (V, \mathcal{A})$; both $0$ and $0'$ represent the single depot ($0$ origin, $0'$ destination).

- $N$ : set of $n$ customers to be served, with $n = |N| \geq 2$.

- $N_0 = N \cup \{0\}$, $N'_0 = N \cup \{0'\}$.

- $\mathcal{A} = \{(0,j)\mid j \in N\} \cup \{(i,j)\mid i,j \in N : i \neq j\} \cup \{(i,0')\mid i \in N\}$ : arc set.

## Parameters

- $t^T_{ij}$ : time for the truck to traverse arc $(i,j) \in \mathcal{A}$.

- $t^D_{ij}$ : time for the drone to traverse arc $(i,j) \in \mathcal{A}$. (In general $t^D_{ij} \leq t^T_{ij}$, but this is not required by the model.)

- $M$ : a sufficiently large constant (big-M).

## Decision Variables

- $x^T_{ij} \in \{0,1\}$ : $1$ if the truck traverses arc $(i,j) \in \mathcal{A}$ (regardless of whether the drone is on-board or airborne).

- $x^D_{ij} \in \{0,1\}$ : $1$ if the drone traverses arc $(i,j) \in \mathcal{A}$ (regardless of whether it is on-board or airborne).

- $y^T_i \in \{0,1\}$ : $1$ if customer $i \in N$ is a truck customer.

- $y^D_i \in \{0,1\}$ : $1$ if customer $i \in N$ is a drone customer.

- $y^C_i \in \{0,1\}$ : $1$ if customer $i \in N$ is a combined customer.

- $a_i \in \mathbb{R}_+$ : arrival time of the truck or the drone (or both) at node $i \in V$.

## Objective

$$\begin{align}
t^* = \min \quad & a_{0'} \tag{1a}
\end{align}$$

## Constraints

$$\begin{align}
\text{s.t.}\quad
& \sum_{(i,j) \in \mathcal{A}} x^T_{ij} = \sum_{(j,i) \in \mathcal{A}} x^T_{ji}
    && \forall\, i \in N, \tag{1b}\\
& \sum_{(i,j) \in \mathcal{A}} x^T_{ij} = y^T_i + y^C_i
    && \forall\, i \in N, \tag{1c}\\
& \sum_{(0,j) \in \mathcal{A}} x^T_{0j} = \sum_{(i,0') \in \mathcal{A}} x^T_{i0'} = 1,
    && \tag{1d}\\
& \sum_{(i,j) \in \mathcal{A}} x^D_{ij} = \sum_{(j,i) \in \mathcal{A}} x^D_{ji}
    && \forall\, i \in N, \tag{1e}\\
& \sum_{(i,j) \in \mathcal{A}} x^D_{ij} = y^D_i + y^C_i
    && \forall\, i \in N, \tag{1f}\\
& \sum_{(0,j) \in \mathcal{A}} x^D_{0j} = \sum_{(i,0') \in \mathcal{A}} x^D_{i0'} = 1,
    && \tag{1g}\\
& y^T_i + y^D_i + y^C_i = 1
    && \forall\, i \in N, \tag{1h}\\
& a_i + t^T_{ij} \leq a_j + M\bigl(1 - x^T_{ij}\bigr)
    && \forall\, (i,j) \in \mathcal{A}, \tag{1i}\\
& a_i + t^D_{ij} \leq a_j + M\bigl(1 - x^D_{ij}\bigr)
    && \forall\, (i,j) \in \mathcal{A}, \tag{1j}\\
& x^D_{ij} + x^D_{ji} \leq y^C_i + y^C_j
    && \forall\, i,j \in N : i < j, \tag{1k}\\
& x^D_{0i} + x^D_{i0'} \leq 1
    && \forall\, i \in N, \tag{1l}\\
& x^T_{ij},\, x^D_{ij} \in \{0,1\}
    && \forall\, (i,j) \in \mathcal{A}, \tag{1m}\\
& y^T_i,\, y^D_i,\, y^C_i \in \{0,1\}
    && \forall\, i \in N, \tag{1n}\\
& a_i \in \mathbb{R}_+
    && \forall\, i \in V. \tag{1o}
\end{align}$$
