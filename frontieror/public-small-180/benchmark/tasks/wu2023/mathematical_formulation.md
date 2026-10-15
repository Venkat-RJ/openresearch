# Original Formulation: Optimal Scenario Reduction for One-Stage Robust Optimization (Cont)

*Source: “Optimal Scenario Reduction for One- and Two-Stage Robust Optimization”, Marc Goerigk and Mohammad Khosravi, 2022.*

The paper frames the conditions of Corollary 3 (one-stage approximation guarantee) as an optimization problem that searches for a reduced uncertainty set $\mathcal{C} = \{\hat{\mathbf c}^1,\dots,\hat{\mathbf c}^K\}$ minimizing the approximation guarantee for any underlying one-stage robust problem. This continuous, nonlinear (bilinear) problem (3)–(9), which the paper refers to as *Cont*, is the first and canonical statement of the problem; its notation ($\lambda$, $\mu$, $t$, $\hat{\mathbf c}^k$) is carried into every subsequent variant, heuristic, and experiment (the “Cont” curve in Figure 3). The integer variants IP-$\mu$, IP-$\lambda$, and the two-stage model are derived afterwards and are collected under **Variants**.

## Sets and Indices

- $[n] = \{1,\dots,n\}$ : components (dimensions) of the cost vectors; index $j \in [n]$.

- $[N] = \{1,\dots,N\}$ : original scenarios in the uncertainty set $\mathcal{U}$; indices $i,\ell \in [N]$.

- $[K] = \{1,\dots,K\}$ : reduced / aggregated scenarios in $\mathcal{C}$; index $k \in [K]$.

## Parameters

- $\mathbf{c}^i \in \mathbb{R}^n_+$ : the $i$-th scenario cost vector, $i \in [N]$; the uncertainty set is $\mathcal{U} = \{\mathbf{c}^1,\dots,\mathbf{c}^N\}$.

- $c^i_j$ : the $j$-th component of scenario $\mathbf{c}^i$ (all values non-negative).

- $K$ : desired number of reduced (aggregated) scenarios, $K < N$.

## Decision Variables

- $t \in \mathbb{R}_+$ : approximation guarantee to be maximized ($1/t$ is the resulting approximation ratio).

- $\lambda_{ki} \in \mathbb{R}_+$ : weight of original scenario $i$ in building aggregated scenario $k$ (a convex combination), $k \in [K],\, i \in [N]$.

- $\mu_{ik} \in \mathbb{R}_+$ : weight of the convex combination of $\mathcal{C}$ that dominates scenario $\mathbf{c}^i$, $i \in [N],\, k \in [K]$.

The aggregated scenarios of $\mathcal{C}$ are recovered as $\displaystyle \hat{c}^k_j = \sum_{i \in [N]} \lambda_{ki}\, c^i_j$.

In the released solution contract, every IP-$\mu$ result includes the complete
binary assignment and the sparse $\lambda$ weights (omitted weights are zero).
These fields are the feasibility certificate used to recompute the reported
$t$; a value-only IP-$\mu$ record is not a valid solution.

## Objective

$$\begin{align}
  \max \quad & t \tag{3}
\end{align}$$

## Constraints

$$\begin{align}
  \text{s.t.}\quad
  & t\, c^i_j \;\leq\; \sum_{k \in [K]} \sum_{\ell \in [N]} \mu_{ik}\, \lambda_{k\ell}\, c^\ell_j
      && \forall\, i \in [N],\, j \in [n] \tag{4}\\[2pt]
  & \sum_{i \in [N]} \lambda_{ki} = 1
      && \forall\, k \in [K] \tag{5}\\
  & \sum_{k \in [K]} \mu_{ik} = 1
      && \forall\, i \in [N] \tag{6}\\
  & t \geq 0 \tag{7}\\
  & \lambda_{ki} \geq 0
      && \forall\, k \in [K],\, i \in [N] \tag{8}\\
  & \mu_{ik} \geq 0
      && \forall\, i \in [N],\, k \in [K] \tag{9}
\end{align}$$
