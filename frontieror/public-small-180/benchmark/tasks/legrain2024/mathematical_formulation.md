# Original Formulation: Personalized Nurse Scheduling Problem (NSP)

*Source: A dedicated pricing algorithm to solve a large family of nurse scheduling problems with branch-and-price, Antoine Legrain & Jérémy Omer, 2024 (INFORMS Journal on Computing).*

## Sets and Indices

- $\mathcal{N}$ : set of nurses, indexed by $i$.

- $\mathcal{D} = \{1,\ldots,n_{\mathcal{D}}\}$ : set of days of the planning horizon, indexed by $d$; $n_{\mathcal{D}}$ is the number of days.

- $\mathcal{S}$ : set of shifts, indexed by $s$. $\mathcal{S}$ includes a special $\mathrm{None}$ (rest) shift; $(d,\mathrm{None})$ is a day off and $(d,s)$ with $s\neq\mathrm{None}$ is a working day.

- $\Sigma$ : set of skills, indexed by $\sigma$.

- An *assignment* is a pair $(d,s)\in\mathcal{D}\times\mathcal{S}$. A generic assignment is denoted $p\in\mathcal{D}\times\mathcal{S}$.

- $n_{\mathcal{W}}$ : number of weekends over the horizon.

- $\Omega_i$ : set of all feasible rosters for nurse $i\in\mathcal{N}$. Each roster $j\in\Omega_i$ is characterized by a binary vector $a^j$ with $a^j_{d,s}=1$ iff roster $j$ includes assignment $(d,s)$.

## Parameters

- $\mathbf{c_x}$ : vector of roster costs ($c_j$ = cost of roster $j\in\Omega_i$; aggregates the soft individual-roster constraint penalties).

- $\mathbf{c_y}$ : vector of penalties for skill-assignment violations ($c_y=0$ when the nurse possesses the assigned skill).

- $\mathbf{c_z}$ : vector of penalties for unsatisfied demands.

- $b_{p,\sigma}$ : minimum demand on assignment $p\in\mathcal{D}\times\mathcal{S}$ for skill $\sigma\in\Sigma$.

- $a^j_p$ : binary parameter, $1$ iff roster $j$ covers assignment $p\in\mathcal{D}\times\mathcal{S}$.

## Decision Variables

- $x_{ij}\in\{0,1\}$ : $1$ iff roster $j\in\Omega_i$ is assigned to nurse $i\in\mathcal{N}$.

- $y_{i,p,\sigma}\in\{0,1\}$ : $1$ iff nurse $i\in\mathcal{N}$ performs skill $\sigma\in\Sigma$ on assignment $p\in\mathcal{D}\times\mathcal{S}$.

- $z_{p\sigma}\ge 0$ : unsatisfied demand for skill $\sigma\in\Sigma$ on assignment $p\in\mathcal{D}\times\mathcal{S}$.

## Objective

$$\begin{align}
\min \quad & \mathbf{c_x}^{T}\mathbf{x} + \mathbf{c_y}^{T}\mathbf{y}
            + \mathbf{c_z}^{T}\mathbf{z} \tag{1a}
\end{align}$$

## Constraints

$$\begin{align}
& \sum_{j\in\Omega_i} x_{ij} = 1,
  && \forall i\in\mathcal{N}, && [\alpha_i] \tag{1b}\\
& \sum_{j\in\Omega_i} a^j_p\, x_{ij} = \sum_{\sigma\in\Sigma} y_{i,p,\sigma},
  && \forall i\in\mathcal{N},\ \forall p\in\mathcal{D}\times\mathcal{S},
  && [\beta_{i,p}] \tag{1c}\\
& \sum_{i\in\mathcal{N}} y_{i,p,\sigma} + z_{p\sigma} \ge b_{p,\sigma},
  && \forall p\in\mathcal{D}\times\mathcal{S},\ \forall \sigma\in\Sigma
  && \tag{1d}\\
& x_{ij},\, y_{ip\sigma} \in \{0,1\},
  && \forall i,j,p,\sigma && \tag{1e}\\
& \mathbf{w},\, \mathbf{z} \ge 0
  && && \tag{1f}
\end{align}$$

## Released executable certificate

The public direct MILP expands each roster into one shift choice for every
nurse-day and emits the parallel complete skill witness. Instance flags select
hard versus soft skill possession and forbidden patterns. If
`demand_equality=true`, (1d) is released as equality with zero slack;
otherwise it remains the stated lower bound. Weekend records are data-driven
and may include Friday. The checker requires exact nurse/day key coverage in
both maps and independently recomputes every objective component.
