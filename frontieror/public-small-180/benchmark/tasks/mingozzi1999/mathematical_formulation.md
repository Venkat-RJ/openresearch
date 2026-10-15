# Original Formulation (Eqs. 3–9)

## Sets and Indices

- $G = (V, A)$: directed graph with $V = \{0\} \cup L \cup B$.

- $L = \{1, \ldots, n\}$: linehaul customers; $B = \{n+1, \ldots, n+m\}$: backhaul customers; vertex $0$ is the depot.

- $L_0 = L \cup \{0\}$; $B_0 = B \cup \{0\}$.

- $A_L = \{(i,j) \in A : i, j \in L_0\}$; $A_B = \{(i,j) \in A : i, j \in B_0\}$; $A_0 = \{(i,j) \in A : i \in L,\ j \in B_0\}$.

- $\mathcal{L}$: set of all feasible linehaul paths (starting at 0 in $G_L$).

- $\mathcal{B}$: set of all feasible backhaul paths (ending at 0 in $G_B$).

- $\mathcal{L}_i \subseteq \mathcal{L}$: linehaul paths passing through customer $i \in L$.

- $\mathcal{L}_i^E \subseteq \mathcal{L}$: linehaul paths ending at customer $i \in L$.

- $\mathcal{B}_j \subseteq \mathcal{B}$: backhaul paths passing through customer $j \in B$.

- $\mathcal{B}_j^S \subseteq \mathcal{B}$: backhaul paths starting at customer $j \in B$.

## Parameters

- $d_{ij} \geq 0$: cost of arc $(i,j) \in A$.

- $q_i \in \mathbb{Z}_+$: demand of customer $i \in L \cup B$.

- $Q$: vehicle capacity (identical for all vehicles).

- $M \geq \max\{M_L, M_B\}$: number of identical vehicles, with $M_L, M_B$ the minimum numbers of vehicles needed to serve linehaul (resp. backhaul) customers.

- $c_\ell$: cost of path $\ell \in \mathcal{L} \cup \mathcal{B}$ (sum of its arc costs).

## Decision Variables

- $x_\ell \in \{0, 1\}$, $\ell \in \mathcal{L}$: 1 if linehaul path $\ell$ is used.

- $y_\ell \in \{0, 1\}$, $\ell \in \mathcal{B}$: 1 if backhaul path $\ell$ is used.

- $\xi_{ij} \in \{0, 1\}$, $(i,j) \in A_0$: 1 if the connector arc $(i,j)$ is used.

## Objective and Constraints

$$\begin{align}
\min \quad & z(IP) = \sum_{\ell \in \mathcal{L}} c_\ell\, x_\ell + \sum_{\ell \in \mathcal{B}} c_\ell\, y_\ell
  + \sum_{(i,j) \in A_0} d_{ij}\, \xi_{ij}
  \tag{3} \\
\text{s.t.}\quad & \sum_{\ell \in \mathcal{L}_i} x_\ell = 1
  && \forall i \in L, \tag{4} \\
& \sum_{\ell \in \mathcal{B}_j} y_\ell = 1
  && \forall j \in B, \tag{5} \\
& \sum_{\ell \in \mathcal{L}_i^E} x_\ell - \sum_{j \in B_0} \xi_{ij} = 0
  && \forall i \in L, \tag{6} \\
& \sum_{\ell \in \mathcal{B}_j^S} y_\ell - \sum_{i \in L} \xi_{ij} = 0
  && \forall j \in B, \tag{7} \\
& \sum_{(i,j) \in A_0} \xi_{ij} = M,
  \tag{8} \\
& x_\ell \in \{0,1\}\ \forall \ell \in \mathcal{L},\quad
  y_\ell \in \{0,1\}\ \forall \ell \in \mathcal{B},\quad
  \xi_{ij} \in \{0,1\}\ \forall (i,j) \in A_0. \tag{9}
\end{align}$$
