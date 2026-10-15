# Original Formulation: Energy Systems Planning Problem

Jacobson, Pecci, Sepulveda, Xu, Jenkins. “A computationally efficient Benders decomposition for energy systems planning problems with detailed operations and time-coupling constraints.” Section 2 (compact form) and Appendix B (detailed constraints).

## Sets and Indices

- $G$: resources (generators, storage units); $G^{UC}$: units subject to unit commitment; $G^{NONRET}$: non-retirable; $G^{RPS}$: RPS-qualifying; $G^{STOR}$: storage; $G^{HYDRO}$: hydro.

- $Z$: spatial zones; $S$: demand segments; $L$: transmission lines.

- $W$: subperiods; $H_w = \{(w-1)\delta_w + 1,\dots,w\delta_w\}$: hours in subperiod $w$; $t_w^0$, $t_w$: first/last hours; $H_w^0 = H_w\setminus\{t_w^0\}$.

- $I_z^{in}$, $I_z^{out}$: lines into/out of zone $z$; $G_z$, $G_z^{STOR}$: resources/storage in zone $z$.

## Decision Variables

**Investment:** $y_g^P$ (capacity MW), $y_g^E$ (storage MWh), $y_l^F$ (line MW), $y_g^{P,\text{NEW}}$, $y_g^{E,\text{NEW}}$, $y_l^{F,\text{NEW}}$ (investments; integer in MILP), $y_g^{P,\text{RET}}$, $y_g^{E,\text{RET}}$ (retirements; integer in MILP).

**Operational (per timestep $t$):** $x_{g,t}^{\text{inj}}$ (injection), $x_{g,t}^{\text{wdw}}$ (storage withdrawal), $x_{s,z,t}^{\text{nse}}$ (non-served energy), $x_{g,t}^{\text{soc}}$ (state of charge), $x_{g,t}^{\text{lvl}}$ (hydro level), $x_{g,t}^{\text{spill}}$ (spillage), $x_{l,t}^{\text{flow}}$ (line flow, free), $x_{g,t}^{\text{commit}}$, $x_{g,t}^{\text{start}}$, $x_{g,t}^{\text{shut}}$ (UC).

**Policy slack:** $x_w^{RPS}$, $x_w^{CO2} \geq 0$.

## Compact Formulation (Section 2, Problem 1)

Let $\mathbf{y}\in\mathbb{R}^m$ group all investment variables and $\mathbf{x}_w\in\mathbb{R}^n$ group all operational variables for subperiod $w$. Matrices $(\mathbf{A}_w,\mathbf{B}_w,\mathbf{b}_w)$ encode operational constraints (Appendix B.2, eqs. 15–24); $(\mathbf{Q}_w,\mathbf{e})$ encode time-coupling policy constraints (eqs. 26–27); $(\mathbf{R},\mathbf{r})$ encode investment-only constraints (eqs. 12–14). Vector $\mathbf{c}_I$ encodes fixed-cost objective terms (eq. 29), and $\mathbf{c}_w$ encodes operational-cost terms (eqs. 30–33).

$$\begin{align}
\min \quad & \mathbf{c}_I^\top \mathbf{y} + \sum_{w\in W} \mathbf{c}_w^\top \mathbf{x}_w \tag{1a} \\
\text{s.t.}\quad
  & \mathbf{A}_w \mathbf{x}_w + \mathbf{B}_w \mathbf{y} \leq \mathbf{b}_w, &&\forall w \in W \tag{1b} \\
  & \sum_{w\in W} \mathbf{Q}_w \mathbf{x}_w \leq \mathbf{e} \tag{1c} \\
  & \mathbf{R}\mathbf{y} \leq \mathbf{r} \tag{1d} \\
  & \mathbf{x}_w \geq 0, &&\forall w \in W \tag{1e} \\
  & \mathbf{y} \geq 0 \tag{1f} \\
  & \mathbf{y} \in \mathbb{Z}^m \tag{1g}
\end{align}$$

## Objective Components (eqs. 29–34)

$$\begin{align}
& c^{\text{FIXED}} = \sum_{g\in G} c_g^{P,\text{INV}} \gamma_g^{P,\text{SIZE}} y_g^{P,\text{NEW}}
  + \sum_{g\in G^{\text{STOR}}} c_g^{E,\text{INV}} \gamma_g^{E,\text{SIZE}} y_g^{E,\text{NEW}}
  \notag\\
& \quad + \sum_{g\in G^{\text{HYDRO}}} c_g^{E,\text{INV}} \rho_g y_g^{P,\text{NEW}}
  + \sum_{g\in G} c_g^{P,\text{FOM}} y_g^P
  + \sum_{g\in G^{\text{STOR}}} c_g^{E,\text{FOM}} y_g^E
  \notag\\
& \quad + \sum_{g\in G^{\text{HYDRO}}} c_g^{E,\text{FOM}} \rho_g y_g^P
  + \sum_{l\in L} c_l^{F,\text{INV}} y_l^{F,\text{NEW}} \tag{29} \\
& c^{\text{VAR}} = \sum_{w,\,t\in H_w}\!\!\Big(\sum_{g\in G} c_g^{\text{VAR}} \alpha_t x_{g,t}^{\text{inj}}
  + \sum_{g\in G^{\text{STOR}}} c_g^{\text{VAR}} \alpha_t x_{g,t}^{\text{wdw}}\Big) \tag{30} \\
& c^{\text{NSE}} = \sum_{w,\,t\in H_w}\sum_{z\in Z}\sum_{s\in S} c_{s,z}^{\text{NSE}} \alpha_t x_{s,z,t}^{\text{nse}} \tag{31} \\
& c^{\text{START}} = \sum_{w,\,t\in H_w}\sum_{g\in G^{UC}} c_g^{\text{START}} \alpha_t x_{g,t}^{\text{start}} \tag{32} \\
& c^{\text{RPS}} + c^{\text{CO2}} = \sum_{w\in W} c^{\text{RPS}} x_w^{RPS}
  + \sum_{w\in W} c^{\text{CO2}} x_w^{CO2} \tag{33} \\
& \text{Objective:}\ c^{\text{FIXED}}+c^{\text{VAR}}+c^{\text{NSE}}+c^{\text{START}}+c^{\text{RPS}}+c^{\text{CO2}} \tag{34}
\end{align}$$

## Constraint Groups (Appendix B)

**B.1 Investment (eqs. 12–14).** Capacity, retirement, and transmission-expansion bounds: $\gamma_g^{P,\text{SIZE}} y_g^{P,\text{NEW}} \leq \overline{y}_g^P$; $\gamma_g^{P,\text{SIZE}} y_g^{P,\text{RET}} \leq y_g^{P,0}$; $y_g^{P,\text{RET}}=0$ for $g\in G^{NONRET}$; $y_g^P = y_g^{P,0} + \gamma_g^{P,\text{SIZE}}(y_g^{P,\text{NEW}}-y_g^{P,\text{RET}})$; analogous for storage ($y_g^E$) with duration limits $\underline{\rho}_g y_g^P \leq y_g^E \leq \overline{\rho}_g y_g^P$; $y_l^{F,\text{NEW}} \leq \overline{y}_l^F$ and $y_l^F = y_l^{F,0} + y_l^{F,\text{NEW}}$.

**B.2 Operations (eqs. 15–24).** (15) Power balance: $$\begin{align*}
\sum_{g\in G_z} x_{g,t}^{\text{inj}} - \sum_{g\in G_z^{\text{STOR}}} x_{g,t}^{\text{wdw}}
  - \sum_{l\in I_z^{\text{out}}} x_{l,t}^{\text{flow}} + \sum_{l\in I_z^{\text{in}}} x_{l,t}^{\text{flow}}
  + \sum_{s\in S} x_{s,z,t}^{\text{nse}} = d_{z,t}.
\end{align*}$$ (16) Capacity/storage limits; (17) minimum output; (18) transmission flow bounds $-y_l^F\leq x_{l,t}^{\text{flow}}\leq y_l^F$; (19) NSE limit $x_{s,z,t}^{\text{nse}}\leq\bar\nu_s d_{z,t}$; (20) storage/hydro state-of-charge dynamics with circular indexing; (21) ramp limits for non-UC resources; (22) unit-commitment logic $x_{g,t}^{\text{commit}}-x_{g,t-1}^{\text{commit}}=x_{g,t}^{\text{start}}-x_{g,t}^{\text{shut}}$; (23) UC ramp limits; (24) minimum up/down time (circular within $H_w$).

**B.3 Policy (eqs. 26–28).** RPS: $$\begin{align*}
\sum_{w\in W}\Big(\sum_{t\in H_w}\sum_{g\in G^{\text{RPS}}} \alpha_t x_{g,t}^{\text{inj}} + x_w^{RPS}\Big)
  \geq \underline{\epsilon}^{\text{RPS}} \sum_{w,\,t,\,z} \alpha_t d_{z,t}.
\end{align*}$$ CO$_2$ cap: $$\begin{align*}
\sum_{w\in W}\Big(\sum_{t\in H_w}\big(\sum_{g\in G} \alpha_t \epsilon_g^{\text{CO2}} x_{g,t}^{\text{inj}}
  + \sum_{g\in G^{\text{STOR}}} \alpha_t \epsilon_g^{\text{CO2}} x_{g,t}^{\text{wdw}}\big) - x_w^{CO2}\Big)
  \leq \overline{\epsilon}^{\text{CO2}}.
\end{align*}$$

## Variable Domains

Investment variables $y_g^{P,\text{NEW}}, y_g^{E,\text{NEW}}, y_g^{P,\text{RET}}, y_g^{E,\text{RET}},
y_l^{F,\text{NEW}} \in \mathbb{Z}_{\geq 0}$ (MILP); $y_g^P, y_g^E, y_l^F \geq 0$. Operational variables non-negative, except $x_{l,t}^{\text{flow}}$ free; UC variables relaxed to $\mathbb{R}_{\geq 0}$. Policy slacks $x_w^{RPS}, x_w^{CO2} \geq 0$.
