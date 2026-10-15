# Original Formulation: Dial-a-Ride Problem (DARP), Per-Vehicle Model

*Source: A Branch-and-Cut Algorithm for the Dial-a-Ride Problem, Cordeau, 2006.*

## Sets and Parameters

- $n$: number of users (requests).

- $G = (N, A)$: complete directed graph with $N = P \cup D \cup \{0, 2n+1\}$.

- $P = \{1, \dots, n\}$: pick-up nodes; $D = \{n+1, \dots, 2n\}$: drop-off nodes; node $0$ is the origin depot and $2n+1$ the destination depot. User $i \in P$ has origin $i$ and destination $n+i$.

- $K$: set of vehicles.

- $Q_k$: capacity of vehicle $k \in K$.

- $T_k$: maximum route duration for vehicle $k \in K$.

- $q_i$: load at node $i$; $q_0 = q_{2n+1} = 0$ and $q_i = -q_{n+i}$ for $i = 1,\dots,n$.

- $d_i \geq 0$: service duration at node $i$; $d_0 = d_{2n+1} = 0$.

- $[e_i, l_i]$: time window at node $i$.

- $c^{k}_{ij}$: routing cost on arc $(i,j)$ (potentially vehicle-dependent).

- $t_{ij}$: travel time on arc $(i,j)$.

- $L$: maximum ride time of a user.

## Decision Variables (Per-Vehicle)

- $x^{k}_{ij} \in \{0,1\}$, $(i,j) \in A$, $k \in K$: $=1$ iff vehicle $k$ traverses arc $(i,j)$.

- $B^{k}_{i} \geq 0$, $i \in N$, $k \in K$: time at which vehicle $k$ begins service at node $i$.

- $Q^{k}_{i} \geq 0$, $i \in N$, $k \in K$: load of vehicle $k$ after visiting node $i$.

- $L^{k}_{i} \geq 0$, $i \in P$, $k \in K$: ride time of user $i$ on vehicle $k$.

## Objective

$$\begin{equation}
\min \;\; \sum_{k \in K} \sum_{i \in N} \sum_{j \in N} c^{k}_{ij}\, x^{k}_{ij} \tag{1}
\end{equation}$$

## Constraints

$$\begin{align}
\sum_{k \in K} \sum_{j \in N} x^{k}_{ij} &= 1, & \forall i \in P \tag{2} \\[2pt]
\sum_{j \in N} x^{k}_{ij} \;-\; \sum_{j \in N} x^{k}_{\,n+i,\,j} &= 0, & \forall i \in P,\; k \in K \tag{3} \\[2pt]
\sum_{j \in N} x^{k}_{0 j} &= 1, & \forall k \in K \tag{4} \\[2pt]
\sum_{j \in N} x^{k}_{j i} \;-\; \sum_{j \in N} x^{k}_{i j} &= 0, & \forall i \in P \cup D,\; k \in K \tag{5} \\[2pt]
\sum_{i \in N} x^{k}_{i,\, 2n+1} &= 1, & \forall k \in K \tag{6} \\[2pt]
B^{k}_{j} &\;\geq\; \bigl(B^{k}_{i} + d_{i} + t_{ij}\bigr)\, x^{k}_{ij}, & \forall i, j \in N,\; k \in K \tag{7} \\[2pt]
Q^{k}_{j} &\;\geq\; \bigl(Q^{k}_{i} + q_{j}\bigr)\, x^{k}_{ij}, & \forall i, j \in N,\; k \in K \tag{8} \\[2pt]
L^{k}_{i} &\;=\; B^{k}_{n+i} - \bigl(B^{k}_{i} + d_{i}\bigr), & \forall i \in P,\; k \in K \tag{9} \\[2pt]
B^{k}_{2n+1} - B^{k}_{0} &\;\leq\; T_{k}, & \forall k \in K \tag{10} \\[2pt]
e_{i} \;\leq\; B^{k}_{i} &\;\leq\; l_{i}, & \forall i \in N,\; k \in K \tag{11} \\[2pt]
t_{i,\,n+i} \;\leq\; L^{k}_{i} &\;\leq\; L, & \forall i \in P,\; k \in K \tag{12} \\[2pt]
\max\{0,\, q_i\} \;\leq\; Q^{k}_{i} &\;\leq\; \min\{Q_k,\, Q_k + q_i\}, & \forall i \in N,\; k \in K \tag{13} \\[2pt]
x^{k}_{ij} &\;\in\; \{0,1\}, & \forall i, j \in N,\; k \in K \tag{14}
\end{align}$$

Constraints (7) and (8) are bilinear in their original form; they may be linearized via standard big-$M$ constraints.
