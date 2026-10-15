# Original Formulation: Capacitated Hub Location Problem with Single Assignment (CHLPSA), Formulation (MP)

*Source: Branch and Price for Large-Scale Capacitated Hub Location Problems with Single Assignment, Contreras, Díaz, and Fernández, 2011.*

## Sets and Parameters

- $G = (N, A)$: complete digraph; $N = \{1,\dots,n\}$ is the set of nodes (potential hub locations).

- Indices $(i,j) \in N \times N$ denote origin/destination pairs; $(k,m) \in N \times N$ denote hub pairs.

- $W_{ij}$: flow with origin $i$ and destination $j$.

- $d_{ij}$: distance from $i$ to $j$ (satisfies the triangle inequality).

- $F_{ijkm} = W_{ij}(\chi\, d_{ik} + \alpha\, d_{km} + \delta\, d_{mj})$: cost of routing $W_{ij}$ along path $i \to k \to m \to j$, where $\chi,\alpha,\delta$ are collection, transfer, and distribution coefficients.

- $f_k$: fixed setup cost of locating a hub at node $k$.

- $b_k$: capacity of hub $k$ (upper bound on total incoming flow).

- $O_i = \sum_{j \in N} W_{ij}$: total outgoing flow at node $i$.

- $D = \sum_{i \in N} O_i$: total flow in the graph.

## Decision Variables

- $z_{ik} \in \{0,1\}$, $i,k \in N$: $=1$ iff node $i$ is assigned to hub $k$; $z_{kk} = 1$ indicates a hub is located at $k$.

- $x_{ijkm} \geq 0$, $i,j,k,m \in N$: $=1$ iff the flow from $i$ to $j$ is routed via hubs $k$ and $m$ (integrality is implied by the binary $z$ variables together with (4) and (5)).

## Objective

$$\begin{equation}
\min \;\; \sum_{k \in N} f_k\, z_{kk}
         \;+\; \sum_{i \in N} \sum_{j \in N} \sum_{k \in N} \sum_{m \in N} F_{ijkm}\, x_{ijkm}
\tag{1}
\end{equation}$$

## Constraints

$$\begin{align}
\sum_{k \in N} \sum_{m \in N} x_{ijkm} &= 1, & \forall\, i, j \in N \tag{2} \\[2pt]
z_{ik} &\;\leq\; z_{kk}, & \forall\, i, k \in N \tag{3} \\[2pt]
\sum_{m \in N} x_{ijkm} &= z_{ik}, & \forall\, i, j, k \in N \tag{4} \\[2pt]
\sum_{k \in N} x_{ijkm} &= z_{jm}, & \forall\, i, j, m \in N \tag{5} \\[2pt]
\sum_{i \in N} O_i\, z_{ik} &\;\leq\; b_k\, z_{kk}, & \forall\, k \in N \tag{6} \\[2pt]
\sum_{k \in N} b_k\, z_{kk} &\;\geq\; D, \tag{7} \\[2pt]
x_{ijkm} &\;\geq\; 0, & \forall\, i, j, k, m \in N \tag{8} \\[2pt]
z_{ik} &\;\in\; \{0,1\}, & \forall\, i, k \in N \tag{9}
\end{align}$$
