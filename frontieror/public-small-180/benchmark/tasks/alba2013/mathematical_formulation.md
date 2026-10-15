# Original Formulation: Double Traveling Salesman Problem with Multiple Stacks (DTSPMS)

*Source: “A Branch-and-Cut Algorithm for the Double Traveling Salesman Problem with Multiple Stacks,” Alba Martı́nez, Cordeau, Dell’Amico, and Iori, INFORMS Journal on Computing, 2013.*

## Sets and Parameters

- $n$: number of customer requests.

- $s$: number of stacks in the vehicle; $l$: maximum height of each stack.

- $G^P = (V^P, A^P)$, $G^D = (V^D, A^D)$: complete directed graphs for the pickup and delivery regions.

- $V^T = \{0^T\} \cup V^T_0$ for $T \in \{P,D\}$, where $0^T$ is the depot, $V^P_0 = \{1^P,\ldots,n^P\}$, $V^D_0 = \{1^D,\ldots,n^D\}$.

- Each request $i \in \{1,\ldots,n\}$ is associated with pickup vertex $i^P$ and delivery vertex $i^D$.

- $c^T_{ij} \ge 0$: routing cost of arc $(i,j) \in A^T$, $T \in \{P,D\}$; $c^T$ from $0^P$ to $0^D$ is zero.

## Decision Variables

- $x^T_{ij} \in \{0,1\}$, for $(i,j) \in A^T$, $T \in \{P,D\}$: equals $1$ iff vertex $j^T$ is visited immediately after $i^T$.

## Objective

$$\begin{equation}
\min \sum_{T \in \{P,D\}} \sum_{(i,j) \in A^T} c^T_{ij}\, x^T_{ij} \tag{1}
\end{equation}$$

## Constraints

$$\begin{align}
\sum_{j \in V^T} x^T_{ij} &= 1 & i \in V^T,\; T \in \{P,D\} \tag{2} \\
\sum_{i \in V^T} x^T_{ij} &= 1 & j \in V^T,\; T \in \{P,D\} \tag{3} \\
\sum_{i \in S} \sum_{j \in S} x^T_{ij} &\le |S| - 1 & S \subsetneq V^T,\; |S| \ge 2,\; T \in \{P,D\} \tag{4} \\
x^T_{ij} &\in \{0,1\} & (i,j) \in A^T,\; T \in \{P,D\} \tag{5}
\end{align}$$

## Infeasible Path (Loading) Constraint

For any load-infeasible pair of paths $(Q,F)$ with $Q = \{p_1,\ldots,p_q\} \subseteq V^P_0$ and $F = \{d_1,\ldots,d_f\} \subseteq V^D_0$: $$\begin{equation}
\sum_{j=1}^{q-1} x^P_{p_j, p_{j+1}} + \sum_{j=1}^{f-1} x^D_{d_j, d_{j+1}} \le |A(Q)| + |A(F)| - 1 \tag{6}
\end{equation}$$
