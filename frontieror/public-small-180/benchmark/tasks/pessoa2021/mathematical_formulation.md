# Original Formulation: Robust Capacitated Vehicle Routing Problem with Knapsack Uncertainty (Robust CVRP)

*Source: Branch-cut-and-price for the robust capacitated vehicle routing problem with knapsack uncertainty, Artur Alves Pessoa, Michael Poss, Ruslan Sadykov, François Vanderbeck, 2021 (Operations Research 69(3), 739–754).*

The problem the paper studies is the robust counterpart of the CVRP. Following the authors’ development (Section 2 reduces the robust feasibility set to a union of deterministic ones; Section 3.2 then states the path/set-partitioning model), the formulation that *defines* the studied problem and is carried into the branch-cut-and-price algorithm and the experiments is the robust path formulation (14)–(17). It is an exponential-column (set-partitioning) model with one route variable per feasible robust route, organized by “vehicle type” $\theta$.

## Sets and Indices

- $G=(V,A)$: complete digraph with node set $V=\{0,1,\dots,n\}$ and arc set $A=\{(i,j)\in V\times V : i\neq j\}$. Node $0$ is the unique depot.

- $V^0=V\setminus\{0\}=\{1,\dots,n\}$: set of customers (written $V_0$ in the tagged equations).

- $R^0$: set of all routes in $G$ starting and ending at the depot (a route visits each other node at most once).

- $V_1,\dots,V_s$: a partition of $\{1,\dots,n\}$ defining the knapsack uncertainty set; $k(i)$ denotes the unique index $k$ with $i\in V_k$.

- $\mathcal{D}$: the knapsack demand-uncertainty set (defined below).

- $\Theta$: finite set of dual vectors $\theta\in\mathbb{R}_+^s$ indexing the deterministic subproblems, given by Eq. (6): $\Theta=\bigl(\{0\}\cup\{\theta^i_{k(i)}\mid i\in V_1\}\bigr)\times\cdots\times
                   \bigl(\{0\}\cup\{\theta^i_{k(i)}\mid i\in V_s\}\bigr)$, with $\theta^0_k=0$ and $\theta^i_{k(i)}=\hat{d}_i/w_i$.

- $\Theta^*\subseteq\Theta$: any set large enough to contain a minimizer of the function minimized in Eq. (4) for each $y\in Y^0$ (e.g. $\Theta$ itself, or $\Theta^{card}$ for $\mathcal{D}^{card}$).

- $\bar{\Theta}\equiv\{\theta\in\Theta^* : \tilde{\mathcal{R}}_\theta\neq\emptyset\}$: the vehicle types surviving the preprocessing test (those whose tightened route set is non-empty). This is the index set actually summed over in the formulation.

- For each $\theta\in\bar{\Theta}$, the robust-feasible route set for type $\theta$: $$\mathcal{R}_\theta\equiv\Bigl\{\, r\in R^0 \ \Big|\ \textstyle\sum_{i\in r} d_i^\theta \le C-b^\top\theta \,\Bigr\}.$$

## Parameters

- $n$: number of customers; $m$: number of homogeneous vehicles.

- $C\in\mathbb{R}_+$: vehicle capacity.

- $c_{ij}\in\mathbb{R}_+$: transportation cost of arc $(i,j)\in A$.

- $\bar{d}_i\in\mathbb{R}_+$: nominal demand of customer $i\in V^0$.

- $\hat{d}_i\in\mathbb{R}_+$: maximum demand deviation of customer $i\in V^0$.

- $w'\in\mathbb{R}_+^n$, $b'\in\mathbb{R}_+^s$: weights and budgets defining $\mathcal{D}$.

- Derived (Section 2): $w_i=w'_i\hat{d}_i$ for $i=1,\dots,n$; $b_k=b'_k-\sum_{i\in V_k} w'_i\bar{d}_i$ for $k=1,\dots,s$.

- For each $\theta\in\Theta$, the type-$\theta$ demand vector $d_i^\theta=\bar{d}_i+\max\{0,\ \hat{d}_i-w_i\theta_{k(i)}\}$, $i=1,\dots,n$, and the type-$\theta$ effective capacity $C-b^\top\theta$.

- $c_r$: cost of route $r$ (sum of arc costs $c_{ij}$ along $r$).

- $a_i^r\in\{0,1\}$: $1$ iff customer $i$ is visited by route $r$.

Demand-uncertainty set (general knapsack form, Section 2.1): $$\mathcal{D}=\Bigl\{\, d\in[\bar{d},\,\bar{d}+\hat{d}] \ \Big|\ \textstyle\sum_{i\in V_k} w'_i d_i \le b'_k,\ k=1,\dots,s \,\Bigr\}.$$ The set of *robust routes* is then $\mathcal{R}\equiv\bigl\{ r\in R^0 \mid \sum_{i\in r} d_i \le C,\ \forall d\in\mathcal{D}\bigr\}$, which by Theorem 3 / Eq. (13) decomposes as $\mathcal{R}=\bigcup_{\theta\in\bar{\Theta}}\mathcal{R}_\theta$.

## Decision Variables

$$\lambda_r\in\mathbb{Z}_+ : \text{number of times route } r \text{ is used,}
  \quad \text{for } \theta\in\bar{\Theta},\ r\in\mathcal{R}_\theta .$$ The model has an exponential number of such route variables.

## Objective

$$\begin{equation}
  \min\ \sum_{\theta\in\bar{\Theta}}\ \sum_{r\in\mathcal{R}_\theta} c_r\,\lambda_r \tag{14}
\end{equation}$$

## Constraints

$$\begin{align}
  \text{s.t.}\quad
  & \sum_{\theta\in\bar{\Theta}}\ \sum_{r\in\mathcal{R}_\theta} a_i^r\,\lambda_r = 1,
    && i\in V_0, \tag{15}\\[4pt]
  & \sum_{\theta\in\bar{\Theta}}\ \sum_{r\in\mathcal{R}_\theta} \lambda_r = m, \tag{16}\\[4pt]
  & \lambda_r\in\mathbb{Z}_+,
    && \theta\in\bar{\Theta},\ r\in\mathcal{R}_\theta. \tag{17}
\end{align}$$ Constraints (15) ensure each customer is served by exactly one route; constraint (16) fixes the number of used vehicles (routes) to $m$.
