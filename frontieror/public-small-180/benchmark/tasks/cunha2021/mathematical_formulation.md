# Original Formulation: Uncapacitated Three-Level Lot-Sizing and Replenishment Problem with a Distribution Structure (3LSPD-U)

*Source: Valid inequalities, preprocessing, and an effective heuristic for the uncapacitated three-level lot-sizing and replenishment problem with a distribution structure, Jesus O. Cunha and Rafael A. Melo, 2021.*

## Sets and Indices

- $F = P \cup W \cup R$: set of all facilities.

- $P = \{p\}$: singleton set containing the production plant (level 0).

- $W$: set of warehouses (level 1).

- $R$: set of retailers (level 2).

- $T = \{1, \ldots, NT\}$: time horizon (planning periods).

- $\delta(i)$: set of successors of facility $i$. For the plant, $\delta(p) = W$; for a warehouse $w \in W$, $\delta(w) \subseteq R$ is the predefined set of retailers it attends.

- $\delta_w(r) \in W$: the predefined unique warehouse associated with retailer $r \in R$.

## Parameters

- $sc^i_t$: fixed setup cost incurred at facility $i \in F$ whenever production/transportation occurs in period $t \in T$.

- $hc^i_t$: per-unit holding cost charged when items are held in inventory at facility $i \in F$ at the end of period $t \in T$.

- $d^r_t$: demand of retailer $r \in R$ in period $t \in T$.

- Derived demands for $i \in P \cup W$: $$d^i_t =
    \begin{cases}
    \sum_{r \in R} d^r_t, & \text{if } i = p; \\
    \sum_{r \in \delta(i)} d^r_t, & \text{if } i \in W.
    \end{cases}$$

- $d^i_{kt} = \sum_{l=k}^{t} d^i_l$: cumulative demand from period $k$ up to $t$ for facility $i \in F$, for $1 \le k \le t \le |T|$.

## Decision Variables

- $x^i_t \geq 0$: amount produced at the plant (if $i = p$) in period $t$, or amount transported to facility $i \in W \cup R$ from its predecessor in period $t$.

- $s^i_t \geq 0$: amount of stock at facility $i \in F$ at the end of period $t \in T$.

- $y^i_t \in \{0,1\}$: setup variable, equal to $1$ whenever $x^i_t > 0$, for $i \in F$, $t \in T$.

## Objective

$$\begin{equation}
z_{STD} = \min \sum_{t \in T} \left( \sum_{i \in F} sc^i_t y^i_t + \sum_{i \in F} hc^i_t s^i_t \right) \tag{1}
\end{equation}$$

## Constraints

$$\begin{align}
& s^i_{t-1} + x^i_t = \sum_{j \in \delta(i)} x^j_t + s^i_t, && \text{for } i \in P \cup W,\ t \in T, \tag{2} \\
& s^r_{t-1} + x^r_t = d^r_t + s^r_t, && \text{for } r \in R,\ t \in T, \tag{3} \\
& x^i_t \leq d^i_{t|T|}\, y^i_t, && \text{for } i \in F,\ t \in T, \tag{4} \\
& x^i_t,\ s^i_t \geq 0, && \text{for } i \in F,\ t \in T, \tag{5} \\
& y^i_t \in \{0,1\}, && \text{for } i \in F,\ t \in T. \tag{6}
\end{align}$$
