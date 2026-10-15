# Original Formulation: Partial Set Covering Location Problem (PSCLP) and Maximal Covering Location Problem (MCLP)

*Source: Benders Decomposition for Very Large Scale Partial Set Covering and Maximal Covering Location Problems, Jean-François Cordeau, Fabio Furini, and Ivana Ljubić, 2019.*

## Sets and Parameters

- $I$: set of potential facility locations, indexed by $i$.

- $J$: set of customers (demand points), indexed by $j$.

- $I(j) \subseteq I$: subset of facilities that can cover customer $j$ (i.e., those within the coverage radius $\hat{R}$ of $j$).

- $J(i) \subseteq J$: subset of customers that can be covered by facility $i$.

- $f_i \ge 0$: opening cost of facility $i \in I$.

- $d_j \ge 0$: demand of customer $j \in J$.

- $D > 0$: minimum total demand that must be covered (PSCLP parameter).

- $B > 0$: budget available for opening facilities (MCLP parameter).

## Decision Variables

- $y_i \in \{0,1\}$, $i \in I$: 1 if facility $i$ is open, 0 otherwise.

- $z_j \in \{0,1\}$, $j \in J$: 1 if customer $j$ is covered by at least one open facility, 0 otherwise.

## 3.1 The Partial Set Covering Location Problem (PSCLP)

### Objective

$$\begin{align}
\min \quad & \sum_{i \in I} f_i\, y_i \tag{1}
\end{align}$$

### Constraints

$$\begin{align}
\sum_{i \in I(j)} y_i &\ge z_j, & \forall j \in J, \tag{2}\\
\sum_{j \in J} d_j\, z_j &\ge D, \tag{3}\\
y_i &\in \{0,1\}, & \forall i \in I, \tag{4}\\
z_j &\in \{0,1\}, & \forall j \in J. \tag{5}
\end{align}$$

The objective (1) minimizes the total cost of open facilities. Constraints (2) are linking constraints: customer $j$ can be counted as covered only if at least one facility in $I(j)$ is open. Constraint (3) forces the total covered demand to be at least $D$. Constraints (4)–(5) are binary restrictions.

## 3.2 The Maximal Covering Location Problem (MCLP)

### Objective

$$\begin{align}
\max \quad & \sum_{j \in J} d_j\, z_j \tag{6}
\end{align}$$

### Constraints

$$\begin{align}
\sum_{i \in I} f_i\, y_i &\le B, \tag{7}\\
\sum_{i \in I(j)} y_i &\ge z_j, & \forall j \in J, \tag{2}\\
y_i &\in \{0,1\}, & \forall i \in I, \tag{4}\\
z_j &\in \{0,1\}, & \forall j \in J. \tag{5}
\end{align}$$

The objective (6) maximizes the total covered customer demand. The knapsack-like constraint (7) ensures that the available budget $B$ for opening the facilities is not exceeded. The remaining constraints (2),(4),(5) are the same as for the PSCLP.
