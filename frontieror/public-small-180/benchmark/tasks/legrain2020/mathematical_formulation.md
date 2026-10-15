# Original Formulation: Nurse Scheduling Problem (NSP)

*Source: A rotation-based branch-and-price approach for the nurse scheduling problem, Antoine Legrain, Jérémy Omer, Samuel Rosat, Mathematical Programming Computation, 12(3), 417–450, 2020.*

This is Formulation (1) of Section 3.3 (“Integer Programming formulation”): the rotation-based IP that the paper writes first as the definition of the NSP it studies. The problem is described narratively in Section 2; the only explicit mathematical formulation is (1), and its notation ($x,\mathbf{w},y,z$ and the rostering-graph flow constraints) is carried into the branch-and-price algorithm of Section 4. The decision variable $x_{ij}$ is indexed over the (exponential) set of feasible rotations/arcs $\Omega_i$; this exponential cardinality is preserved (the paper generates rotations by column generation, but that is an algorithmic detail of Section 4, not part of the problem definition).

## Sets and Indices

- $\mathcal{N}$ : set of nurses, indexed by $i \in \{1,\dots,|\mathcal{N}|\}$.

- $K = 7M$ : number of days in the planning horizon ($M$ weeks), indexed by $k \in \{1,\dots,K\}$.

- $\mathcal{S}$ : set of shifts, indexed by $s$.

- $\Sigma$ : set of skills, indexed by $\sigma$.

- $p = (k,s)$ : an *assignment* (worked shift $s$ on day $k$); ranges over $\{1,\dots,K\}\times\mathcal{S}$.

- $\Omega_i$ : set of feasible rotations for nurse $i$ (exponential cardinality).

- $\mathcal{V}_i$ : set of vertices in the rostering graph of nurse $i$; $o_i$ source node, $d_i$ sink node.

- $\mathcal{A}^+_{iv}$, $\mathcal{A}^-_{iv}$ : arcs entering / leaving vertex $v$ in the rostering graph of nurse $i$ (each arc $j$ corresponds to a rotation or a resting period).

- $\mathcal{N}_\sigma$ : set of nurses that can perform skill $\sigma$; $\Sigma_i$ : set of skills nurse $i$ can perform.

## Parameters

- $D_{p\sigma}$ : minimum demand in nurses performing skill $\sigma$ on assignment $p$.

- $O_{p\sigma}$ : optimal demand in nurses performing skill $\sigma$ on assignment $p$.

- $\mathbf{c}$ : vector of the cost of each arc in the rotation graph of each nurse (encodes penalties for **S2,S3,S4,S5**).

- $\mathbf{c}_w = (c_{S6}, c_{S6}, c_{S7})$ : vector of unit costs for violations of **S6**–**S7**.

- $c_{S1}$ : unit penalty for insufficient staffing for optimal coverage (**S1**).

- $\mathbf{a}_j$ : Boolean vector indicating which assignment $p$ is in rotation $j$ (component $a_{jp}$).

- $\mathbf{b}_j$ : vector containing the number of assignments and worked weekends in rotation $j$ of nurse $i$.

- $\mathbf{B}_i = (L^-_i, L^+_i, WE^+_i)$ : vector of the bounds associated with **S6**–**S7** (min/max total worked days and max worked weekends for nurse $i$).

## Decision Variables

- $x_{ij} \in \{0,1\}$ : $=1$ iff the rotation (or resting period) associated with arc $j$ is part of the roster of nurse $i$.

- $\mathbf{w}_i \in \mathbb{N}^3$ : vector of the violations of the soft constraints **S6**–**S7** in the roster of nurse $i$.

- $y_{ip\sigma} \in \{0,1\}$ : $=1$ iff nurse $i$ performs skill $\sigma$ on assignment $p$.

- $z_{p\sigma} \in \mathbb{N}$ : violation of the optimal demand constraint for assignment $p$ and skill $\sigma$.

## Objective

$$\begin{align}
\min \quad
  & \underbrace{\mathbf{c}^{T}\mathbf{x}}_{\textbf{S2,S3,S4,S5}}
  + \underbrace{\sum_{i} \mathbf{c}_{w}^{T}\mathbf{w}_i}_{\textbf{S6,S7}}
  + \underbrace{c_{S1}\sum_{l,\sigma} z_{l\sigma}}_{\textbf{S1}}
  \tag{1a}
\end{align}$$

## Constraints

$$\begin{align}
\textbf{[H1]}\quad
  & \sum_{j \in \mathcal{A}^+_{io_i}} x_{ij} = 1,
  && \forall i && [\alpha_{io_i}] \tag{1b}\\[2pt]
\textbf{[H1,H3]}\quad
  & \sum_{j \in \mathcal{A}^+_{iv}} x_{ij} - \sum_{j \in \mathcal{A}^-_{iv}} x_{ij} = 0,
  && \forall i,\ \forall v \in \mathcal{V}_i \setminus \{o_i,d_i\} && [\alpha_{iv}] \tag{1c}\\[2pt]
\textbf{[S6,S7]}\quad
  & \sum_{j \in \Omega_i} \mathbf{b}_j\, x_{ij} + \mathbf{w}_i \le \mathbf{B}_i,
  && \forall i && [\boldsymbol{\beta}_i] \tag{1d}\\[2pt]
\textbf{[H4]}\quad
  & \sum_{j \in \Omega_i}\sum_{p} a_{jp}\, x_{ij} - \sum_{\sigma \in \Sigma_i} y_{ip\sigma} = 0,
  && \forall i,\ p && [\delta_{ip}] \tag{1e}\\[2pt]
\textbf{[H2]}\quad
  & \sum_{i \in \mathcal{N}_\sigma} y_{ip\sigma} \ge D_{p\sigma},
  && \forall p,\ \sigma && \tag{1f}\\[2pt]
\textbf{[S1]}\quad
  & \sum_{i \in \mathcal{N}_\sigma} y_{ip\sigma} + z_{p\sigma} \ge O_{p\sigma},
  && \forall p,\ \sigma && \tag{1g}\\[2pt]
  & x_{ij},\, y_{ip\sigma} \in \{0,1\},
  && \forall i,\ j,\ p,\ \sigma && \tag{1h}\\[2pt]
  & \mathbf{w},\, \mathbf{z} \ge 0. && && \tag{1i}
\end{align}$$
