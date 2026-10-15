# Sets and Indices

- $V$: set of incoming vessels, indexed by $i, j$.

- $K$: set of available subblocks in the yard, indexed by $k$.

- $B$: set of berths in the quay, indexed by $b$.

- $P_i$: set of Quay Crane (QC) profiles for vessel $i$, indexed by $p$.

- $H$: number of time steps in the planning horizon.

- $E = \max_{i \in V,\, p \in P_i} h_{ip}$: maximum handling time across vessels and profiles.

- $T = \{1, \ldots, H+E\}$: extended time-step set, indexed by $t$.

# Parameters

- $h_{ip}$: handling time of vessel $i$ under QC profile $p \in P_i$ (in time steps).

- $q_{ipm}$: number of QCs used by profile $p$ at the $m$-th step of the handling operation of vessel $i$, $m = 1,\ldots,h_{ip}$.

- $Q_t$: maximum number of QCs available at time step $t$.

- $[a^f_i, b^f_i]$: feasible service time window for vessel $i$.

- $[a^e_i, b^e_i]$: expected service time window for vessel $i$.

- $r_i$: number of subblocks to be reserved for vessel $i$.

- $l_i, u_i$: number of containers to be loaded, unloaded for vessel $i$.

- $D^L_{kb}$: length of loading route between subblock $k$ and berth $b$.

- $D^U_b$: average length of unloading route from berth $b$ to all subblocks.

- $c^p_i$: per-step penalty-cost coefficient for deviation from the expected service window.

- $c^o$: global operation-cost coefficient per unit of route length.

- $M$: sufficiently large positive constant.

# Decision Variables

- $\omega_{ib} \in \{0,1\}$: $1$ if berth $b$ is allocated to vessel $i$.

- $\delta_{ijb} \in \{0,1\}$: $1$ if both vessels $i$ and $j$ dwell at berth $b$ and $i$ is served before $j$.

- $\varphi_{ik} \in \{0,1\}$: $1$ if subblock $k$ is reserved for vessel $i$.

- $\gamma_{ip} \in \{0,1\}$: $1$ if vessel $i$ is served by QC profile $p$.

- $\mu_{it} \in \{0,1\}$: $1$ if vessel $i$ begins handling at time step $t$.

- $\eta_{ipt} \in \{0,1\}$: $1$ if vessel $i$ is served by profile $p$ AND begins at time step $t$.

- $\alpha_i, \beta_i \in T$: integer start and end time steps of handling for vessel $i$.

- $\varrho_b, \varsigma_b \in T$: integer start and end time steps of berth $b$’s occupancy window.

- $\sigma_t \ge 0$ (integer): number of QCs used at time step $t$.

# Model \[M1\] — Original Nonlinear Formulation

## Objective (1)

$$\begin{align}
  \min \Biggl\{
    & \sum_{i \in V} c^p_i\, \bigl[(a^e_i - \alpha_i)^+ + (\beta_i - b^e_i)^+\bigr] \notag \\
    & \;+\; c^o \sum_{i \in V}\sum_{b \in B}\sum_{k \in K} \omega_{ib}\, \varphi_{ik}\, D^L_{kb}\, (l_i / r_i) \notag \\
    & \;+\; c^o \sum_{i \in V}\sum_{b \in B} \omega_{ib}\, D^U_b\, u_i
  \Biggr\} \tag{1}
\end{align}$$ The first term is a piecewise-linear (absolute-value-like) penalty on deviation from the expected service window; the second term contains bilinear products $\omega_{ib}\,\varphi_{ik}$.

## Constraints

$$\begin{align}
  & \sum_{i \in V} \varphi_{ik} \le 1, && \forall\, k \in K \tag{2} \\
  & \sum_{k \in K} \varphi_{ik} = r_i, && \forall\, i \in V \tag{3} \\
  & \sum_{p \in P_i} \gamma_{ip} = 1, && \forall\, i \in V \tag{4} \\
  & \sum_{b \in B} \omega_{ib} = 1, && \forall\, i \in V \tag{5} \\
  & \sum_{t=1}^{H} \mu_{it} = 1, && \forall\, i \in V \tag{6} \\
  & \sum_{t \in T} t\, \mu_{it} = \alpha_i, && \forall\, i \in V \tag{7} \\
  & \alpha_i + \sum_{p \in P_i} \gamma_{ip}\, h_{ip} - 1 = \beta_i, && \forall\, i \in V \tag{8} \\
  & \alpha_i + \sum_{p \in P_i} \gamma_{ip}\, h_{ip} \le \alpha_j + (1 - \delta_{ijb})\, M, && \forall\, i, j \in V,\ i \neq j,\ b \in B \tag{9} \\
  & \delta_{ijb} + \delta_{jib} \le \omega_{ib}, && \forall\, i, j \in V,\ i \neq j,\ b \in B \tag{10} \\
  & \delta_{ijb} + \delta_{jib} \ge \omega_{ib} + \omega_{jb} - 1, && \forall\, i, j \in V,\ i \neq j,\ b \in B \tag{11} \\
  & \alpha_i \ge a^f_i, && \forall\, i \in V \tag{12} \\
  & \beta_i \le b^f_i, && \forall\, i \in V \tag{13} \\
  & \eta_{ipt} \ge \gamma_{ip} + \mu_{it} - 1, && \forall\, i \in V,\ p \in P_i,\ t \in T \tag{14} \\
  & \sigma_t = \sum_{i \in V}\sum_{p \in P_i} \sum_{m=\max\{1,\,t-h_{ip}+1\}}^{t} \eta_{ipm}\, q_{ip\,(t-m+1)}, && \forall\, t \in T \tag{15} \\
  & \sigma_t \le Q_t, && \forall\, t \in \{E+1, \ldots, H\} \tag{16} \\
  & \sigma_t + \sigma_{t+H} \le Q_t, && \forall\, t \in \{1, \ldots, E\} \tag{17} \\
  & \varrho_b \le \alpha_i + (1 - \omega_{ib})\, M, && \forall\, i \in V,\ b \in B \tag{18} \\
  & \varsigma_b \ge \beta_i + (\omega_{ib} - 1)\, M, && \forall\, i \in V,\ b \in B \tag{19} \\
  & \varsigma_b - \varrho_b \le H - 1, && \forall\, b \in B \tag{20}
\end{align}$$

## Variable Domains (21)–(28)

$$\begin{align}
  & \omega_{ib} \in \{0,1\}, && \forall\, i \in V,\ b \in B \tag{21} \\
  & \delta_{ijb} \in \{0,1\}, && \forall\, i, j \in V,\ i \neq j,\ b \in B \tag{22} \\
  & \varphi_{ik} \in \{0,1\}, && \forall\, i \in V,\ k \in K \tag{23} \\
  & \gamma_{ip} \in \{0,1\}, && \forall\, i \in V,\ p \in P_i \tag{24} \\
  & \mu_{it} \in \{0,1\}, && \forall\, i \in V,\ t \in T \tag{25} \\
  & \eta_{ipt} \in \{0,1\}, && \forall\, i \in V,\ p \in P_i,\ t \in T \tag{26} \\
  & \sigma_t \ge 0,\ \text{integer}, && \forall\, t \in T \tag{27} \\
  & \varrho_b, \varsigma_b \in T, && \forall\, b \in B \tag{28}
\end{align}$$

# Linearization \[M2\] (Same Formulation Section)

In Section 4.3 the paper introduces auxiliary variables $\tau^{a+}_i, \tau^{a-}_i, \tau^{b+}_i, \tau^{b-}_i \ge 0$ for the penalty terms (constraints (29)–(31)) and $\theta_{ikb} \in \{0,1\}$ satisfying $\theta_{ikb} \ge \omega_{ib} + \varphi_{ik} - 1$ (constraints (32)–(33)) to linearize the bilinear product $\omega_{ib}\,\varphi_{ik}$. The linearized objective (34) is $$\begin{align}
  \min \Biggl\{
    & \sum_{i \in V} c^p_i (\tau^{a+}_i + \tau^{b+}_i)
    + c^o \sum_{i \in V}\sum_{b \in B}\sum_{k \in K} \theta_{ikb}\, D^L_{kb}\, (l_i/r_i) \notag \\
    & + c^o \sum_{i \in V}\sum_{b \in B} \omega_{ib}\, D^U_b\, u_i
  \Biggr\} \tag{34}
\end{align}$$ subject to (2)–(33).
