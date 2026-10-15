# Original Formulation (Eqs. 2–19)

## Sets and Indices

- $N$: set of all ships berthing at any of the ports; $N^* \subseteq N$ are ships being optimized; $\bar{N} \subseteq N$ are external (fixed) ships.

- $P$: set of ports; $L_p$: length of quay in port $p \in P$.

- $S$: set of (discretized) sailing speeds.

- $P_i \subseteq P$: ports planned to be visited by ship $i \in N^*$, in visiting order.

- $C_i = \{1, \ldots, c_i\}$: port calls for ship $i \in N$; $c_i$ is the last port visit.

- $\rho_i^c \in P$: port corresponding to port visit $c \in C_i$ for ship $i \in N$.

- $N_p \subseteq N$: ships visiting port $p \in P$.

- $C_i^p \subseteq C_i$: port call positions of ship $i$ visiting port $p$.

## Parameters

- $x_0^{i,c}$: ideal berthing position (leftmost) for ship $i \in N^*$, visit $c \in C_i$.

- $EST_i^c,\ EFT_i^c,\ LFT_i^c$: earliest start, expected finish, latest finish times.

- $\beta$: relative increase in handling time per unit distance from the ideal position.

- $\Delta^{p,p'}$: distance between ports $p, p'$; $\Theta_s$: travel time per unit distance at speed $s$.

- $\Gamma_s^i$: fuel consumption per unit distance at speed $s$ for ship $i$.

- $l_i$: length of ship $i$; $h_0^{i,c}$: minimum handling time at the preferred berthing position.

- Cost coefficients: $F$ (fuel, USD/tonne), $H$ (handling, USD/h), $D$ (delay, USD/h), $I$ (waiting, USD/h), $U$ (LFT-exceeding penalty, USD).

- $M$: a sufficiently large constant (big-M).

## Decision Variables

Primary:

- $x_i^c \in \mathbb{R}_+$: leftmost berthing position of ship $i \in N$ at port visit $c$.

- $y_i^c \in \mathbb{R}_+$: start time of berthing of ship $i \in N$ at port visit $c$.

- $v_{i,s}^c \in \{0,1\}$: 1 if ship $i \in N^*$ sails from visit $c$ to $c+1$ at speed $s$.

- $d_i^c \in \mathbb{R}_+$: delay over $EFT_i^c$ for ship $i \in N^*$ at port visit $c$.

- $u_i^c \in \mathbb{R}_+$: delay over $LFT_i^c$ for ship $i \in N^*$ at port visit $c$.

Auxiliary:

- $\sigma_{i,j}^{c,c'} \in \{0,1\}$: 1 if ship $i$ is positioned left of vessel $j$ at port $p$.

- $\delta_{i,j}^{c,c'} \in \{0,1\}$: 1 if ship $i$ finishes berthing before vessel $j$ starts at port $p$.

- $r^{i,c} \in \mathbb{R}_+$: distance between ideal and actual berthing position.

Dependent:

- $a_i^c \in \mathbb{R}_+$: arrival time of ship $i \in N^*$ at port visit $c$.

- $h_i^c \in \mathbb{R}_+$: handling time of ship $i \in N^*$ at port visit $c$.

## Objective and Constraints

$$\begin{align}
\min \quad & \sum_{i \in N^*} \Bigg( \sum_{c \in C_i} \Big( I(y_i^c - a_i^c) + H(h_i^c) + D(d_i^c) + U(u_i^c) \Big)
  + \sum_{c \in C_i \setminus \{c_i\}} F\Big( \textstyle\sum_{s \in S} v_{i,s}^c\, \Gamma_s^i\, \Delta^{\rho_i^c, \rho_i^{c+1}} \Big) \Bigg)
  \tag{2} \\
\text{s.t.}\quad & x_i^c + l_i \leq L_p,
  && \forall\, p \in P,\ i \in N_p,\ c \in C_i^p \tag{3} \\
& x_i^c + l_i \leq x_j^{c'} + L_p\,(1 - \sigma_{i,j}^{c,c'}),
  && \forall\, p \in P,\ i,j \in N_p,\ i \neq j,\ c \in C_i^p,\ c' \in C_j^p \tag{4} \\
& y_i^c + h_i^c \leq y_j^{c'} + M\,(1 - \delta_{i,j}^{c,c'}),
  && \forall\, p \in P,\ i,j \in N_p,\ i \neq j,\ c \in C_i^p,\ c' \in C_j^p \tag{5} \\
& \sigma_{i,j}^{c,c'} + \sigma_{j,i}^{c',c} + \delta_{i,j}^{c,c'} + \delta_{j,i}^{c',c} \geq 1,
  && \forall\, p \in P,\ i,j \in N_p,\ i < j,\ c \in C_i^p,\ c' \in C_j^p,\ c < c' \tag{6} \\
& y_i^c + h_i^c + \sum_{s \in S} v_{i,s}^c\, \Theta_s\, \Delta^{\rho_i^c, \rho_i^{c+1}} = a_i^{c+1},
  && \forall\, i \in N^*,\ c \in C_i \setminus \{c_i\} \tag{7} \\
& a_i^c \leq y_i^c,
  && \forall\, i \in N^*,\ c \in C_i \tag{8} \\
& EST_i^c \leq y_i^c,
  && \forall\, i \in N^*,\ c \in C_i \tag{9} \\
& y_i^c + h_i^c - EFT_i^c \leq d_i^c,
  && \forall\, i \in N^*,\ c \in C_i \tag{10} \\
& y_i^c + h_i^c - LFT_i^c \leq u_i^c,
  && \forall\, i \in N^*,\ c \in C_i \tag{11} \\
& (1 + \beta\, r^{i,c})\, h_0^{i,c} = h_i^c,
  && \forall\, i \in N^*,\ c \in C_i \tag{12} \\
& x_i^c - x_0^{i,c} \leq r^{i,c},
  && \forall\, i \in N^*,\ c \in C_i \tag{13} \\
& x_0^{i,c} - x_i^c \leq r^{i,c},
  && \forall\, i \in N^*,\ c \in C_i \tag{14} \\
& \sum_{s \in S} v_{i,s}^c = 1,
  && \forall\, i \in N^*,\ c \in C_i \setminus \{c_i\} \tag{15} \\
& y_i^c,\ x_i^c \geq 0,
  && \forall\, i \in N,\ c \in C_i \tag{16} \\
& a_i^c,\ h_i^c,\ d_i^c,\ u_i^c,\ r^{i,c} \geq 0,
  && \forall\, i \in N^*,\ c \in C_i \tag{17} \\
& v_{i,s}^c \in \{0,1\},
  && \forall\, i \in N^*,\ c \in C_i \setminus \{c_i\},\ s \in S \tag{18} \\
& \sigma_{i,j}^{c,c'},\ \delta_{i,j}^{c,c'} \in \{0,1\},
  && \forall\, p \in P,\ i,j \in N_p,\ i \neq j,\ c \in C_i^p,\ c' \in C_j^p \tag{19}
\end{align}$$
