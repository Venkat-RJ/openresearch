# Original Formulation: Single-Product Maritime Inventory Routing (IRP)

*Source: A Branch-Price-and-Cut Algorithm for Single-Product Maritime Inventory Routing, F. G. Engineer, K. C. Furman, G. L. Nemhauser, M. W. P. Savelsbergh, J.-H. Song, Operations Research 60(1):106–122, 2012.*

The paper states (§3, “A Column Generation Formulation”) a time-indexed *mixed-integer column generation formulation*, in which integrality is enforced in a nontrivial way. The problem the paper studies is defined by the master problem (1)–(6); voyages (columns) are characterized by the pricing problem (7)–(13). The variables $\lambda^{v,r}$ are indexed over the exponentially large set $R_v$ of all possible voyages and are generated dynamically; the formulation is inherently a decomposition-based formulation.

## Sets and Indices

- $J_S$ : set of supply (load) facilities/ports.

- $J_D$ : set of demand (discharge) facilities/ports.

- $V$ : set of heterogeneous vessels.

- $T$ : length of the planning horizon; $t \in \{1,\ldots,T\}$ : time periods.

- $R_v$ : set of *all possible* voyages for vessel $v \in V$ (exponentially large). A voyage is defined by the sequence of ports visited, the timing, and the quantity loaded/discharged at each port.

- $\mathcal{N}_v = (N_v, A_v)$ : time-expanded network for vessel $v$. A node $n=(j,t)\in N_v$ corresponds to location $j\in J_S\cup J_D$ and time $t\in\{1,\ldots,T\}$. An arc $e=((j_1,t_1),(j_2,t_2))\in A_v$ corresponds to relocation ($j_1\neq j_2$, $t_2-t_1$ = travel time) or demurrage ($j_1=j_2$, $t_2=t_1+1$).

- $n_s,\, n_t$ : source and sink nodes for vessel $v$ (earliest/latest time points the vessel is available for use).

## Parameters

- $b_{j,t}$ : amount produced ($j\in J_S$) or consumed ($j\in J_D$) at facility $j$ during time period $t$.

- $Q_{j,t}$ : storage capacity at facility $j$ during time period $t$.

- $I_{j,0}$ : inventory available at facility $j$ at the start of the horizon.

- $F_j^{\min},\, F_j^{\max}$ : minimum / maximum amount that can be loaded/unloaded at port $j$ in a single time period.

- $Q_v$ : cargo capacity of vessel $v\in V$.

- $c^{v,r}$ : cost of voyage $r\in R_v$ for vessel $v$ (transportation and demurrage costs, plus expense/revenue from procuring/supplying the product). Given parameter of the column.

- $z_{j,t}^{v,r}\in\{0,1\}$ : given $0$-$1$ indicator for the load/discharge decision at port $j$ and time $t$ for voyage $r$ of vessel $v$. Given parameter of the column.

- $f_{j,t}^{v,r}$ : amount of product loaded/discharged at port $j$ and time $t$ for voyage $r$ of vessel $v$, with $f_{j,t}^{v,r}\ge 0$ if $j\in J_S$ and $f_{j,t}^{v,r}\le 0$ if $j\in J_D$. Given parameter of the column.

#### Pricing-problem parameters (vessel $v$, network $\mathcal{N}_v$).

- $\pi_{j,t}$ : dual values for the inventory-balance constraints (1)/(2).

- $\alpha_v$ : dual value for the convexity constraint (4).

- $c_e$ : fixed (transportation/demurrage) cost of arc $e\in A_v$.

- $\bar c_n$ : per-unit cost at node $n=(j,t)$ for procuring/supplying a single unit of product, adjusted by the dual value $\pi_{j,t}$.

- $U_n \le Q_v$ : maximum amount of inventory the vessel can have on board when entering port $j$ at time $t$ (incorporates vessel capacity, draft limits, and end-of-horizon empty-vessel requirements).

- $F_n^{\min},\, F_n^{\max}$ : minimum/maximum load/discharge at node $n$, with $F_n^{\min}=F_j^{\min},\, F_n^{\max}=F_j^{\max}$ if $j\in J_S$, and $F_n^{\min}=-F_j^{\max},\, F_n^{\max}=-F_j^{\min}$ if $j\in J_D$.

## Decision Variables

- $\lambda^{v,r}$ : continuous variable, $0\le \lambda^{v,r}\le 1$, indicating the fraction of voyage $r\in R_v$ selected for vessel $v$.

- $I_{j,t}$ : inventory at port $j$ at time $t$ (continuous).

*Pricing-problem variables (per vessel $v$):*

- $x_e$ : $0$-$1$ arc-selection variable, $e\in A_v$.

- $z_n$ : $0$-$1$ load/discharge decision at node $n\in N_v$.

- $f_n$ : amount loaded/discharged at node $n\in N_v$.

- $F_n$ : amount of inventory on board the vessel when entering node $n\in N_v$.

## Objective

Column generation master problem: $$\begin{equation}
\min \;\; \sum_{v\in V}\sum_{r\in R_v} c^{v,r}\,\lambda^{v,r}. \notag
\end{equation}$$

## Constraints

Master problem: $$\begin{align}
I_{j,t} &= I_{j,t-1} + b_{j,t}
        - \sum_{v\in V}\sum_{r\in R_v} f_{j,t}^{v,r}\,\lambda^{v,r},
        && j\in J_S,\; t=1,\ldots,T, \tag{1}\\
I_{j,t} &= I_{j,t-1} - b_{j,t}
        - \sum_{v\in V}\sum_{r\in R_v} f_{j,t}^{v,r}\,\lambda^{v,r},
        && j\in J_D,\; t=1,\ldots,T, \tag{2}\\
0 &\le I_{j,t} \le Q_{j,t},
        && j\in J_S\cup J_D,\; t=1,\ldots,T, \tag{3}\\
\sum_{r\in R_v}\lambda^{v,r} &= 1, && v\in V, \tag{4}\\
\lambda^{v,r} &\ge 0, && v\in V,\; r\in R_v, \tag{5}\\
\sum_{r\in R_v} z_{j,t}^{v,r}\,\lambda^{v,r} &\in \{0,1\},
        && v\in V,\; j\in J_S\cup J_D,\; t=1,\ldots,T. \tag{6}
\end{align}$$ Constraints (1)–(2) are the inventory-balance constraints for supply and demand ports; (3) ensures inventory never exceeds storage capacity nor is depleted (safety stock may be incorporated as a nonzero lower bound); (4)–(5) allow a convex combination of voyages per vessel; (6) enforces integrality on the load/discharge decisions (not on $\lambda^{v,r}$, which remain continuous).

## Pricing Problem (column generation, for a given vessel $v$)

Given dual values $\pi_{j,t}$ for (1)/(2) and $\alpha_v$ for (4), find a voyage of minimum reduced cost: $$\begin{equation}
\arg\min_{r\in R_v}\;
\left\{\, c^{v,r}
- \Big( \alpha_v + \sum_{j\in J_S\cup J_D}\sum_{t=1}^{T} \pi_{j,t}\,f_{j,t}^{v,r}\Big)
\right\}. \notag
\end{equation}$$ Using the time-expanded network $\mathcal{N}_v=(N_v,A_v)$, the pricing problem for vessel $v$ is stated as: $$\begin{equation}
\min \;\; \sum_{e\in A_v} c_e\, x_e + \sum_{n\in N_v} \bar c_n\, f_n \notag
\end{equation}$$ $$\begin{align}
\sum_{e=(n,n')\in A_v} x_e - \sum_{e=(n',n)\in A_v} x_e
  &= \begin{cases}
       +1 & \text{if } n=n_s,\\
       -1 & \text{if } n=n_t,\\
        0 & \text{if } n\in N_v\setminus\{n_s,n_t\},
     \end{cases} \tag{7}\\
z_n &\le \sum_{e=(n,n')\in A_v} x_e, && n\in N_v, \tag{8}\\
F_n^{\min}\, z_n &\le f_n \le F_n^{\max}\, z_n, && n\in N_v, \tag{9}\\
% Constraint (10) is NONLINEAR (bilinear: product of x_e and (F_{n'}+f_{n'})).
% The paper states it can be linearized using standard "big M" techniques
% (Glover 1975), but does not write the linearized form in this section.
F_n &= \sum_{e=(n',n)\in A_v} x_e\,(F_{n'} + f_{n'}), && n\in N_v, \tag{10}\\
0 &\le F_n \le U_n, && n\in N_v, \tag{11}\\
x_e &\in \{0,1\}, && e\in A_v, \tag{12}\\
z_n &\in \{0,1\}, && n\in N_v. \tag{13}
\end{align}$$ Constraints (7) ensure $x_e$ induces a source-to-sink path in $\mathcal{N}_v$; (8) allows a load/discharge at $n$ only if $n$ is visited; (9) bounds the load/discharge quantity; (10) and (11) track the on-board inventory $F_n$ when entering each node, keeping it between $0$ and $U_n$.
