# Original Formulation: Bi-Objective Single-Source Capacitated Facility Location Problem (BO-SSCFLP)

*Source: “Bi-objective branch-and-cut algorithms based on LP relaxation and bound sets,” Gadegaard, Nielsen, and Ehrgott, submitted to INFORMS Journal on Computing (2019).*

## Sets and Parameters (Reconstruction)

- $\mathcal{I}$: set of potential facilities (indexed by $i$).

- $\mathcal{J}$: set of customers (indexed by $j$).

- $f^1_i, f^2_i$: fixed cost of opening facility $i$ under objectives $1$ and $2$ (integer-valued).

- $c^1_{ij}, c^2_{ij}$: assignment cost of customer $j$ to facility $i$ under objectives $1$ and $2$ (integer-valued).

- $s_i$: capacity of facility $i$; $d_j$: demand of customer $j$.

## Decision Variables (Reconstruction)

- $y_i \in \{0,1\}$: $1$ if facility $i$ is opened.

- $x_{ij} \in \{0,1\}$: $1$ if customer $j$ is single-sourced by facility $i$.

## Objectives (Reconstruction – paper defers to online supplement)

$$\begin{align}
\min \; z_1 &= \sum_{i \in \mathcal{I}} f^1_i\, y_i + \sum_{i \in \mathcal{I}} \sum_{j \in \mathcal{J}} c^1_{ij}\, x_{ij} \tag{1} \\
\min \; z_2 &= \sum_{i \in \mathcal{I}} f^2_i\, y_i + \sum_{i \in \mathcal{I}} \sum_{j \in \mathcal{J}} c^2_{ij}\, x_{ij} \tag{2}
\end{align}$$

## Constraints (Reconstruction)

$$\begin{align}
\sum_{i \in \mathcal{I}} x_{ij} &= 1, & j \in \mathcal{J} \tag{3} \\
\sum_{j \in \mathcal{J}} d_j\, x_{ij} &\le s_i\, y_i, & i \in \mathcal{I} \tag{4} \\
x_{ij} &\le y_i, & i \in \mathcal{I},\; j \in \mathcal{J} \tag{5} \\
x_{ij} \in \{0,1\},\quad y_i &\in \{0,1\}, & i \in \mathcal{I},\; j \in \mathcal{J} \tag{6}
\end{align}$$

Equations (3) are the SOS1 single-sourcing assignment equalities, (4) are knapsack capacity constraints (an indicator-style linking with $y_i$), and (5) are the standard linking inequalities enforcing that a customer may be assigned only to an open facility. Both objectives map into $\mathbb{Z}$ when all cost coefficients are integer-valued, as assumed by the paper.
