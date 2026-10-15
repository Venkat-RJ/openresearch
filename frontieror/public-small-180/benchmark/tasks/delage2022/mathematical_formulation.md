# Original Formulation: Open-Ended Continuous Stock Cutting Problem (Open-Ended CSCP)

*Source: Logic-Based Benders Decomposition for Integrated Process Configuration and Production Planning Problems, Karim Pérez Martı́nez, Yossiri Adulyasak, and Raf Jans, 2022 (original OP formulation of Hajizadeh and Lee, 2007).*

## Sets and Parameters

- $T$: set of product types, indexed by $i$.

- $P$: set of cutting patterns, indexed by $j$.

- $C_r$: time required for a single pattern use (cutting time per repetition).

- $C_s$: setup time for a used pattern.

- $d_i$: demand for product $i$.

- $l_i$: length of product $i$.

- $L$: length of the cutting machine.

- $M$: large number, defined as $M = \max_{i \in T}\{d_i\}$.

## Decision Variables

- $s_j \in \{0,1\}$, $\forall j \in P$: 1 if pattern $j$ is used, 0 otherwise (configuration decision, part of $\mathbf{x}$).

- $a_{ij} \in \mathbb{Z}_+$, $\forall i \in T, j \in P$: number of pieces of product $i$ in pattern $j$ (configuration decision, part of $\mathbf{x}$).

- $z_j \in \mathbb{R}_+$, $\forall j \in P$: number of repetitions (uses) of pattern $j$ (production planning decision, part of $\mathbf{y}$).

- $x_i \in \mathbb{R}_+$, $\forall i \in T$: units of product $i$ produced as extra pieces (production planning decision, part of $\mathbf{y}$).

- $q_{ij} \in \mathbb{R}_+$, $\forall i \in T, j \in P$: units of product $i$ produced by pattern $j$ (production quantities).

## Objective

$$\begin{align}
\text{OP:} \quad \min \quad \mathbf{c}^{\mathsf{T}}\mathbf{x} + \mathbf{d}^{\mathsf{T}}\mathbf{y} \;:\;
\min \quad & C_s \sum_{j \in P} s_j + C_r \sum_{j \in P} z_j \tag{15}
\end{align}$$

## Constraints

$$\begin{align}
\mathbf{Ax} \ge \mathbf{a}: \quad & \sum_{i \in T} l_i\, a_{ij} \le L\, s_j,
  & \forall j \in P, \tag{16}\\
\mathbf{Bx} + \mathbf{Cy} \ge \mathbf{b}: \quad & z_j \le M\, s_j,
  & \forall j \in P, \tag{17}\\
\mathbf{q} = f(\mathbf{x},\mathbf{y}): \quad & q_{ij} = a_{ij}\, z_j,
  & \forall i \in T,\; j \in P, \tag{18}\\
\mathbf{Dy} + \mathbf{Eq} \ge \mathbf{e}: \quad & \sum_{j \in P} q_{ij} + x_i \ge d_i,
  & \forall i \in T, \tag{19}\\
& \sum_{j \in P} z_j \ge \sum_{i \in T} x_i, \tag{20}\\
& z_{j-1} \ge z_j,
  & \forall j \in P : j > 1, \tag{21}\\
\mathbf{x} &\in \mathbb{B}^{|\mathcal{A}|} :\; s_j \in \{0,1\},\; a_{ij} \in \mathbb{Z}_+,
  & \forall i \in T,\; j \in P, \tag{22}\\
\mathbf{y} \in \mathbb{R}_+^{|\mathcal{B}|} \times \mathbb{Z}_+^{|\mathcal{C}|};\;
  \mathbf{q} &\in \mathbb{R}_+^{|\mathcal{D}|} :\; x_i \ge 0,\; z_j \in \mathbb{Z}_+;\; q_{ij} \ge 0,
  & \forall i \in T,\; j \in P. \tag{23}
\end{align}$$

The objective (15) minimizes the total setup and cutting time. Constraints (16) are the configuration (machine length) constraints, which ensure that the total length of the pieces in a used pattern does not exceed the machine length. Constraints (17) link the pattern-use indicator $s_j$ with the number of pattern repetitions $z_j$. **Constraints (18) are the nonlinear (bilinear) production quantity constraints**: the amount of product $i$ produced by pattern $j$ equals the product of the configuration variable $a_{ij}$ (pieces of $i$ per pattern $j$) and the production level $z_j$ (number of repetitions). Constraints (19) ensure demand fulfillment, allowing demand to be met jointly by patterns and extra pieces. Constraints (20) limit the total number of extra pieces by the total number of pattern repetitions. Symmetry-breaking inequalities (21) order the used patterns by the number of repetitions. Constraints (22)–(23) define the variable domains.

## Paper-Supported Variants

The closed-ended CSCP is the special case of (15)-(23) in which no extra pieces can be produced:

$$x_i = 0, \qquad \forall i \in T.$$

The open-ended CSCP with knife-dependent setups adds parameter $C_k$, the setup time per knife, and variable $n_j \ge 0$, the number of knives in pattern $j$. Its objective is

$$\min\; C_s\sum_{j\in P}s_j + C_r\sum_{j\in P}z_j + C_k\sum_{j\in P}n_j,$$

with the additional constraint from Section 1.1.2 of the online supplement,

$$n_j \ge \sum_{i\in T}a_{ij} + M(s_j-1), \qquad \forall j\in P.$$

Because $n_j$ has a positive objective coefficient, it equals the number of pieces cut simultaneously by each used pattern at optimum. All remaining open-ended constraints, including the extra-piece constraints, continue to apply to this variant.
