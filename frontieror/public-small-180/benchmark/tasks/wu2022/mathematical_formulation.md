# Original Formulation: Capacitated Facility Location and Production Planning Problem (CFLPP) — Network Reformulation \[NetR\]

*Source: “A supervised learning-driven heuristic for solving the facility location and production planning problem,” Tao Wu, Lei Huang, Zhe Liang, Xiandong Zhang, Canrong Zhang, European Journal of Operational Research, 301 (2022) 785–796.*

The paper studies the CFLPP. In Section 2 it states the problem only in the compact matrix form $\min\{c_1 X + c_2 Y + c_3 Z : A_1 X + A_2 Y + A_3 Z \ge b\}$, and gives the unique explicit MILP that defines the problem solved throughout the paper as the *network reformulation* \[NetR\] in Appendix A (equations (2)–(10)). The \[NetR\] notation ($Y^k_{jt}, W^k_{jt\ell}, U^k_j, Z^k$) is the one carried into the SLD heuristic and the computational experiments (objective values reported are for this model), so \[NetR\] is the original formulation reproduced below.

## Sets and Indices

- $\{1,\dots,T\}$: planning-horizon periods; indexed by $s,p,\ell,t$. $T$ is the number of periods.

- $\{1,\dots,K\}$: plants; indexed by $k$. $K$ is the number of plants.

- $\{1,\dots,C\}$: customers; indexed by $c$. $C$ is the number of customers.

- $\{1,\dots,J\}$: items; indexed by $j$. $J$ is the number of items.

- $\mathcal{K}_j \subseteq \{1,\dots,K\}$: plants capable of producing item $j$.

- $\mathcal{J}_k \subseteq \{1,\dots,J\}$: items that can be produced at plant $k$.

- $\mathcal{C}_j \subseteq \{1,\dots,C\}$: customers who have ordered item $j$.

## Parameters

- $d^c_{jt}$: demand of item $j$ for customer $c$ in period $t$.

- $td_{jt}$: total demand of item $j$ in period $t$, $td_{jt}=\sum_{c\in\mathcal{C}_j} d^c_{jt}$.

- $sc^k_{jt}$: setup cost for producing a lot of item $j$ at plant $k$ in period $t$.

- $pc^k_{jt}$: production cost per unit of item $j$ at plant $k$ in period $t$.

- $hc^k_{jt}$: per-period inventory-holding cost per unit of item $j$ at plant $k$ in period $t$.

- $st^k_{jt}$: setup time for producing a lot of item $j$ at plant $k$ in period $t$.

- $pt^k_{jt}$: production time per unit of item $j$ at plant $k$ in period $t$.

- $cap^k_t$: capacity of plant $k$ in period $t$.

- $fc^k$: open cost of plant $k$, with $fc^k=\sum_{t\in\{1,\dots,T\}} fc^k_t$ (the $fc^k_t$ are defined in Wu et al. 2020).

- $tc^{kc}_{jt}$: transportation cost for delivering a unit of item $j$ from plant $k$ to customer $c$ in period $t$.

- $sd_{jt\ell}$: total demand of item $j$ in periods $t$ until $\ell$, $sd_{jt\ell}=\sum_{p=t}^{\ell} td_{jp}$.

- $cv^k_{jt\ell}$: total production and inventory-holding cost for producing item $j$ at plant $k$ in period $t$ to satisfy total demand in periods $t$ until $\ell$: $$cv^k_{jt\ell} = pc^k_{jt}\cdot sd_{jt\ell} + \sum_{s=t+1}^{\ell}\sum_{p=t}^{s-1} hc^k_{jp}\cdot td_{js}.$$

## Decision Variables

- $Y^k_{jt}\in\{0,1\}$: setup variable, $=1$ when production is set up for item $j$ at plant $k$ in period $t$.

- $W^k_{jt\ell}\ge 0$: fraction (percentage) of production of item $j$ at plant $k$ in period $t$ used to satisfy the accumulated demand for item $j$ from period $t$ to period $\ell$ ($t\le\ell\le T$). *Continuous.*

- $U^k_j\in\{0,1\}$: plant-location variable, $=1$ when plant $k$ is assigned to produce item $j$.

- $Z^k\in\{0,1\}$: plant-opening variable, $=1$ when plant $k$ is open.

## Objective

$$\begin{equation}
\min\;
\sum_{j=1}^{J}\sum_{t=1}^{T}\sum_{k\in\mathcal{K}_j}
\left(
  sc^k_{jt}\,Y^k_{jt}
  + \sum_{\ell=t}^{T} cv^k_{jt\ell}\,W^k_{jt\ell}
  + \sum_{c\in\mathcal{C}_j} d^c_{jt}\,tc^{kc}_{jt}\,U^k_j
\right)
+ \sum_{k=1}^{K} fc^k\,Z^k
\tag{2}
\end{equation}$$

## Constraints

$$\begin{align}
& \sum_{k\in\mathcal{K}_j} U^k_j = 1,
  && \forall\, j\in\{1,\dots,J\} \tag{3}\\[2pt]
& U^k_j \le Z^k,
  && \forall\, j\in\{1,\dots,J\},\; k\in\mathcal{K}_j \tag{4}\\[2pt]
& Y^k_{jt} \le U^k_j,
  && \forall\, j\in\{1,\dots,J\},\; t\in\{1,\dots,T\},\; k\in\mathcal{K}_j \tag{5}\\[2pt]
& 1 = \sum_{t=1}^{T}\sum_{k\in\mathcal{K}_j} W^k_{j,1,t},
  && \forall\, j\in\{1,\dots,J\} \tag{6}\\[2pt]
& \sum_{\ell=1}^{t-1} W^k_{j,\ell,t-1} = \sum_{\ell=t}^{T} W^k_{jt\ell},
  && \forall\, j\in\{1,\dots,J\},\; t\in\{2,\dots,T\},\; k\in\mathcal{K}_j \tag{7}\\[2pt]
& \sum_{j\in\mathcal{J}_k}\!\left( st^k_{jt}\,Y^k_{jt}
    + \sum_{\ell=t}^{T} pt^k_{jt}\,sd_{jt\ell}\,W^k_{jt\ell}\right)
    \le cap^k_t\,Z^k,
  && \forall\, t\in\{1,\dots,T\},\; k\in\{1,\dots,K\} \tag{8}\\[2pt]
& \sum_{\ell=t}^{T} W^k_{jt\ell} \le Y^k_{jt},
  && \forall\, j\in\{1,\dots,J\},\; t\in\{1,\dots,T\},\; k\in\mathcal{K}_j \tag{9}\\[2pt]
& W^k_{jt\ell}\ge 0,\quad Y^k_{jt},\,U^k_j,\,Z^k\in\{0,1\},
  && \forall\, j\in\{1,\dots,J\},\; t\in\{1,\dots,T\},\; k\in\mathcal{K}_j \tag{10}
\end{align}$$
