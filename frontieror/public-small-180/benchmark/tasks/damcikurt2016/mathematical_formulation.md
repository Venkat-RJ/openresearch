# Original Formulation: Multistage Stochastic Self-Scheduling Unit Commitment (MSS)

*Source: Strong Formulations for the Multistage Stochastic Self-Scheduling Unit Commitment, Kai Pan and Yongpei Guan, 2016.*

## Sets and Parameters

- $\mathcal{T} = (\mathcal{V}, \mathcal{E})$: scenario tree with $T$ time periods.

- $i \in \mathcal{V}$: nodes of the scenario tree (root node is node $0$).

- $t(i)$: time period of node $i$.

- $i^{-}$: unique parent of node $i$ (defined for $i \ne 0$).

- $\mathcal{V}(i)$: set of all descendants of $i$, including $i$ itself.

- $\mathcal{H}_r(i) = \{k \in \mathcal{V}(i) : 0 \le t(k) - t(i) \le r-1\}$: nodes used to express minimum up/down time constraints.

- $p_i$: probability of the state represented by node $i$.

- $q_i$: electricity price (\$/MWh) at node $i$.

- $L, \ell$: minimum-up, minimum-down time (periods).

- $\overline{C}, \underline{C}$: upper/lower generation limit (MW) when online.

- $V^{+}, V^{-}$: ramp-up/ramp-down rate limit (MW/period).

- $\bar{U}, \underline{U}$: start-up and shut-down cost (\$).

- $f(\cdot)$: nondecreasing convex fuel cost function (approximated by a piecewise linear function).

## Decision Variables

- $y_i \in \{0,1\}$, $\forall i \in \mathcal{V}$: 1 if generator is online at node $i$.

- $u_i \in \{0,1\}$, $\forall i \in \mathcal{V}\setminus\{0\}$: 1 if generator starts up at node $i$.

- $x_i \ge 0$, $\forall i \in \mathcal{V}$: electricity generation amount (MW) at node $i$.

Note: the original formulation also introduces $v_i$ (shut-down indicator), but $v_i$ is eliminated in the MSS formulation below via the relation $v_i = y_{i^{-}} - y_i + u_i$.

For the JSON solution certificate, the maps for $x$, $y$, and $u$ use sparse
encoding. A node ID omitted from a map takes the value in the optional
`sparse_defaults` map for that variable (zero when the default is absent), and
an explicitly listed node overrides that default. This is only a serialization
convention and does not alter the full-vector formulation below.

## Objective (MSS, with $v_i$ eliminated)

$$\begin{align}
\max \quad & \sum_{i \in \mathcal{V}} p_i \Bigl( q_i x_i - \bigl(\bar{U}\, u_i + \underline{U}(y_{i^{-}} - y_i + u_i) + f(x_i)\bigr) \Bigr) \tag{1a}
\end{align}$$

## Constraints

$$\begin{align}
y_i - y_{i^{-}} &\le y_k,
  & \forall i \in \mathcal{V}\setminus\{0\},\; \forall k \in \mathcal{H}_L(i), \tag{1b}\\
y_{i^{-}} - y_i &\le 1 - y_k,
  & \forall i \in \mathcal{V}\setminus\{0\},\; \forall k \in \mathcal{H}_\ell(i), \tag{1c}\\
y_i - y_{i^{-}} &\le u_i,
  & \forall i \in \mathcal{V}\setminus\{0\}, \tag{1d}\\
\underline{C}\, y_i \le x_i &\le \overline{C}\, y_i,
  & \forall i \in \mathcal{V}, \tag{1f}\\
x_i - x_{i^{-}} &\le V^{+}\, y_{i^{-}} + \overline{C}(1 - y_{i^{-}}),
  & \forall i \in \mathcal{V}\setminus\{0\}, \tag{1g}\\
x_{i^{-}} - x_i &\le V^{-}\, y_i + \overline{C}(1 - y_i),
  & \forall i \in \mathcal{V}\setminus\{0\}, \tag{1h}\\
u_i &\le \min\{y_i,\; 1 - y_{i^{-}}\},
  & \forall i \in \mathcal{V}\setminus\{0\}, \tag{2}\\
y_i &\in \{0,1\},\; \forall i \in \mathcal{V};\quad u_i \in \{0,1\},\; \forall i \in \mathcal{V}\setminus\{0\}; \quad x_i \ge 0,\; \forall i \in \mathcal{V}. \notag
\end{align}$$

Constraints (1b)–(1c) enforce minimum-up/minimum-down time. Constraint (1d) links the turn-on variable $u_i$ to the transition of $y$. Constraints (1f) bound the generation level when online and fix $x_i = 0$ when offline. Constraints (1g)–(1h) impose ramp-up and ramp-down limits. Constraint (2) ensures $u_i = 1$ only when $y_{i^{-}} = 0$ and $y_i = 1$ (a genuine start-up), removing the spurious configurations left by (1b)–(1d) once $v$ has been substituted out. The feasible region is $$P = \bigl\{(x,y,u) \in \mathbb{R}^{|\mathcal{V}|} \times \mathbb{B}^{|\mathcal{V}|}
  \times \mathbb{B}^{|\mathcal{V}|-1}
  :\; (1b)\text{--}(1d),\; (1f)\text{--}(1h),\; (2)\bigr\}.$$
