# Original Formulation: Pickup and Delivery Problem with Time Windows and LIFO Loading (PDPTWL)

*Source: Branch-Price-and-Cut Algorithms for the Pickup and Delivery Problem with Time Windows and Last-In-First-Out Loading, Cherkesly, Desaulniers, and Laporte, 2015.*

## Sets and Parameters

- $n$: number of requests.

- $G = (N, A)$: directed graph with $N = \{0, 1, \dots, 2n, 2n+1\}$; node $0$ is the origin depot and $2n+1$ the destination depot.

- $P = \{1, \dots, n\}$: set of pickup nodes; $D = \{n+1, \dots, 2n\}$: set of delivery nodes. Request $i \in P$ is paired with delivery node $n+i$.

- $A$: arc set, consisting of (i) $(0,j)$ for $j \in P$; (ii) $(i,2n+1)$ for $i \in D$; (iii) $(i,j)$ with $i \in P$, $j \in P \cup \{n+i\}$; (iv) $(i,j)$ with $i \in D$, $j \in (D \cup P) \setminus \{i-n\}$. In particular, arcs $(i, n+j)$ with $i \in P$, $j \in P \setminus \{i\}$ are *excluded* (they violate LIFO a priori).

- $K$: unrestricted set of identical vehicles.

- For each $i \in P$: $\Omega_i$ is the set of subsets $S \subseteq N$ such that $\{0, 2n+1, i, n+i\} \not\subseteq S$ and there exists a request $j$ with $(j \in S,\, n+j \notin S)$ or $(j \notin S,\, n+j \in S)$.

- $q_i$: load picked up or delivered at $i$; $q_0 = q_{2n+1} = 0$, $q_i > 0$ for $i \in P$, $q_{n+i} = -q_i$ for $i \in P$.

- $s_i$: service duration at $i$ ($s_0 = s_{2n+1} = 0$). $[w_i, \overline{w}_i]$: time window at node $i$.

- $Q$: (identical) vehicle capacity; $c_{ij}$: nonnegative travel cost on arc $(i,j)$; $t_{ij}$: nonnegative travel time on $(i,j)$.

## Decision Variables

- $x^{k}_{ij} \in \{0,1\}$ for $(i,j) \in A$, $k \in K$: $=1$ iff vehicle $k$ uses arc $(i,j)$.

- $T^{k}_{i} \geq 0$ for $i \in N$, $k \in K$: time at which vehicle $k$ begins service at node $i$.

- $Q^{k}_{i} \geq 0$ for $i \in N$, $k \in K$: load of vehicle $k$ upon leaving node $i$.

## Objective

$$\begin{equation}
\min \;\; \sum_{k \in K} \sum_{(i,j) \in A} c_{ij}\, x^{k}_{ij} \tag{1}
\end{equation}$$

## Constraints

$$\begin{align}
\sum_{k \in K} \sum_{j \in N} x^{k}_{ij} &= 1, & \forall i \in P \tag{2} \\[2pt]
\sum_{j \in N} x^{k}_{ij} \;-\; \sum_{j \in N} x^{k}_{\,n+i,\,j} &= 0, & \forall i \in P,\; k \in K \tag{3} \\[2pt]
\sum_{j \in N} x^{k}_{0 j} &= 1, & \forall k \in K \tag{4} \\[2pt]
\sum_{j \in N} x^{k}_{j i} \;-\; \sum_{j \in N} x^{k}_{i j} &= 0, & \forall i \in P \cup D,\; k \in K \tag{5} \\[2pt]
\sum_{i \in N} x^{k}_{i,\, 2n+1} &= 1, & \forall k \in K \tag{6}
\end{align}$$

LIFO constraint (exponential family of infeasible-subset cuts): $$\begin{align}
\sum_{\substack{(i,j) \in A \\ j \in S}} x^{k}_{ij}
 \;+\; \sum_{\substack{(l,j) \in A \\ l,j \in S}} x^{k}_{lj}
 \;+\; \sum_{\substack{(j,\,n+i) \in A \\ j \in S}} x^{k}_{j,\,n+i}
 &\;\leq\; |S|, & \forall S \in \Omega_i,\; i \in P,\; k \in K \tag{7}
\end{align}$$

Load and time constraints (stated in their original nonlinear form): $$\begin{align}
Q^{k}_{j} &\;\geq\; Q^{k}_{i} + q_{j}\, x^{k}_{ij}, & \forall (i,j) \in A,\; k \in K \tag{8} \\[2pt]
\max\{0,\, q_i\} \;\leq\; Q^{k}_{i} &\;\leq\; \min\{Q,\, Q + q_i\}, & \forall i \in N,\; k \in K \tag{9} \\[2pt]
T^{k}_{j} &\;\geq\; T^{k}_{i} + s_{i} + t_{ij}\, x^{k}_{ij}, & \forall (i,j) \in A,\; k \in K \tag{10} \\[2pt]
w_{i} \;\leq\; T^{k}_{i} &\;\leq\; \overline{w}_{i}, & \forall i \in N,\; k \in K \tag{11} \\[2pt]
T^{k}_{i} + t_{i,\,n+i} + s_{i} &\;\leq\; T^{k}_{n+i}, & \forall i \in P \tag{12} \\[2pt]
x^{k}_{ij} &\;\in\; \{0,1\}, & \forall (i,j) \in A,\; k \in K \tag{13}
\end{align}$$

Constraints (8) and (10) are nonlinear (bilinear) and are written as given in the paper; they may be linearized via standard big-$M$ constraints.
