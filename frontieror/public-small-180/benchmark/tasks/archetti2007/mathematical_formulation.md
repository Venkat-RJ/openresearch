# Original Formulation: Vendor-Managed Inventory-Routing with Order-Up-To Level (VMIR-OU)

*Source: A Branch-and-Cut Algorithm for a Vendor-Managed Inventory-Routing Problem, Claudia Archetti, Luca Bertazzi, Gilbert Laporte, Maria Grazia Speranza, 2007.*

## Sets and Parameters

- $\mathcal{M} = \{1, 2, \ldots, n\}$: set of retailers.

- Vertex $0$: the common supplier; $\mathcal{M}' = \mathcal{M} \cup \{0\}$ is the set of all vertices.

- $\mathcal{T} = \{1, \ldots, H\}$: set of discrete time periods (time horizon of length $H$).

- $\mathcal{T}' = \mathcal{T} \cup \{H+1\}$: extended time horizon (the period $H+1$ is included in the inventory-cost accounting).

- $\mathcal{S} \subseteq \mathcal{M}$: a generic subset of retailers (used in the subtour elimination constraints).

- $r_{0t}$: product quantity made available at the supplier at time $t \in \mathcal{T}$, with $r_{00} = 0$.

- $r_{st}$: product quantity consumed by retailer $s \in \mathcal{M}$ at time $t \in \mathcal{T}$, with $r_{s0} = 0$, $s \in \mathcal{M}$.

- $B_0$: starting inventory level at the supplier.

- $U_s$: maximum inventory level at retailer $s \in \mathcal{M}$.

- $I_{s0}$: starting inventory level at retailer $s \in \mathcal{M}$, with $I_{s0} \leq U_s$.

- $h_0$: unit inventory cost at the supplier.

- $h_s$: unit inventory cost at retailer $s \in \mathcal{M}$.

- $C$: vehicle capacity.

- $c_{ij}$: transportation cost from vertex $i$ to vertex $j$, symmetric ($c_{ij} = c_{ji}$), $i, j \in \mathcal{M}'$.

## Decision Variables

- $x_{st} \geq 0$: quantity shipped to retailer $s \in \mathcal{M}$ at time $t \in \mathcal{T}$ (continuous).

- $B_t$: inventory level at the supplier at time $t \in \mathcal{T}'$ (continuous).

- $I_{st}$: inventory level at retailer $s \in \mathcal{M}$ at time $t \in \mathcal{T}'$ (continuous).

- $z_{it} \in \{0,1\}$: equal to $1$ if vertex $i \in \mathcal{M}'$ is visited at time $t \in \mathcal{T}$, $0$ otherwise.

- $y^t_{ij}$: equal to $1$ if $j$ immediately follows $i$ in the route traveled at time $t$, $0$ otherwise; binary for retailer–retailer edges, and in $\{0,1,2\}$ for the retailer–supplier edge ($y^t_{i0} = 2$ means a direct round trip serving only retailer $i$).

## Objective

$$\begin{equation}
\min \quad \sum_{t \in \mathcal{T}'} h_0 B_t
   + \sum_{s \in \mathcal{M}} \sum_{t \in \mathcal{T}'} h_s I_{st}
   + \sum_{i \in \mathcal{M}'} \sum_{\substack{j \in \mathcal{M}' \\ j < i}} \sum_{t \in \mathcal{T}} c_{ij}\, y^t_{ij}
\tag{1}
\end{equation}$$

## Constraints

$$\begin{align}
B_t &= B_{t-1} + r_{0,t-1} - \sum_{s \in \mathcal{M}} x_{s,t-1},
    & & t \in \mathcal{T}' \tag{2} \\
B_t &\geq \sum_{s \in \mathcal{M}} x_{st},
    & & t \in \mathcal{T} \tag{3} \\
I_{st} &= I_{s,t-1} + x_{s,t-1} - r_{s,t-1},
    & & s \in \mathcal{M},\ t \in \mathcal{T}' \tag{4} \\
I_{st} &\geq 0,
    & & s \in \mathcal{M},\ t \in \mathcal{T}' \tag{5} \\
x_{st} &\geq U_s z_{st} - I_{st},
    & & s \in \mathcal{M},\ t \in \mathcal{T} \tag{6} \\
x_{st} &\leq U_s - I_{st},
    & & s \in \mathcal{M},\ t \in \mathcal{T} \tag{7} \\
x_{st} &\leq U_s z_{st},
    & & s \in \mathcal{M},\ t \in \mathcal{T} \tag{8} \\
\sum_{s \in \mathcal{M}} x_{st} &\leq C,
    & & t \in \mathcal{T} \tag{9} \\
\sum_{s \in \mathcal{M}} x_{st} &\leq C z_{0t},
    & & t \in \mathcal{T} \tag{10} \\
\sum_{\substack{j \in \mathcal{M}' \\ j < i}} y^t_{ij} + \sum_{\substack{j \in \mathcal{M}' \\ j > i}} y^t_{ji} &= 2 z_{it},
    & & i \in \mathcal{M}',\ t \in \mathcal{T} \tag{11} \\
\sum_{i \in \mathcal{S}} \sum_{\substack{j \in \mathcal{S} \\ j < i}} y^t_{ij} &\leq \sum_{i \in \mathcal{S}} z_{it} - z_{kt},
    & & \mathcal{S} \subseteq \mathcal{M},\ t \in \mathcal{T},\ \text{for some } k \in \mathcal{S} \tag{12} \\
x_{st} &\geq 0,
    & & s \in \mathcal{M},\ t \in \mathcal{T} \tag{13} \\
y^t_{ij} &\in \{0, 1\},
    & & i \in \mathcal{M},\ j \in \mathcal{M},\ j < i,\ t \in \mathcal{T} \tag{14} \\
y^t_{i0} &\in \{0, 1, 2\},
    & & i \in \mathcal{M},\ t \in \mathcal{T} \tag{15} \\
z_{it} &\in \{0, 1\},
    & & i \in \mathcal{M}',\ t \in \mathcal{T} \tag{16}
\end{align}$$

In constraints (2) and (4), $r_{00} = 0$, $x_{s0} = 0$, and $r_{s0} = 0$ for $s \in \mathcal{M}$.
