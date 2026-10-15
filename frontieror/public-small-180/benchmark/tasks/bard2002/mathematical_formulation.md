# Original Formulation: Vehicle Routing Problem with Time Windows (VRPTW)

*Source: “A Branch-and-Cut Procedure for the Vehicle Routing Problem with Time Windows,” Bard, Kontoravdis, and Yu, Transportation Science 36(2):250–269, 2002.*

## Sets and Parameters

- $I = \{1,\ldots,n\}$: set of customers; $I_0 = I \cup \{0\}$ includes the depot $0$.

- $G_0 = (V, E)$: connected graph with $V = I_0$.

- $q_i > 0$, $\sigma_i \ge 0$, $[a_i, b_i]$: demand, service time, and time window for customer $i \in I$.

- $Q$: vehicle capacity; $\tau_{ij}$: travel time between nodes $i,j \in I_0$ (service time $\sigma_i$ folded in; if $a_j - b_i > \tau_{ij}$, replace $\tau_{ij}$ by $a_j - b_i$).

- Derived: $T_{ij} = b_i - a_j$; $Q_j = Q - q_j$.

## Decision Variables

- $x_{ij} \in \{0,1\}$ for $i,j \in I_0$: $1$ if a vehicle traverses arc $(i,j)$.

- $t_i \in \mathbb{R}$ for $i \in I$: departure time from customer $i$.

- $y_i \in \mathbb{R}$ for $i \in I$: accumulated vehicle load at departure from customer $i$.

## Objective

$$\begin{equation}
(\text{VRPTW}) \quad \min \sum_{i=1}^{n} x_{0i} \tag{1}
\end{equation}$$

## Constraints

$$\begin{align}
\sum_{j \in I_0} x_{ij} &= 1, & i \in I \tag{2} \\
\sum_{j \in I_0} x_{ij} - \sum_{j \in I_0} x_{ji} &= 0, & i \in I_0 \tag{3} \\
t_j &\ge t_i + \tau_{ij} x_{ij} - T_{ij}(1 - x_{ij}), & i,j \in I \tag{4} \\
y_j &\ge y_i + q_j - Q_j(1 - x_{ij}), & i,j \in I \tag{5} \\
q_i \le y_i &\le Q, & i \in I \tag{6} \\
a_i \le t_i &\le b_i, & i \in I \tag{7} \\
x_{ij} &\in \{0,1\}, & i,j \in I_0 \tag{8}
\end{align}$$
