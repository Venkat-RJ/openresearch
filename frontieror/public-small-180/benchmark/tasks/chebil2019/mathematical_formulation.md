# Original Formulation: 0–1 Knapsack Problem with Setups (KPS)

*Source: An exact approach for the 0–1 Knapsack Problem with Setups, Federico Della Croce, Fabio Salassa, Rosario Scatamacchia, 2016 (Computers and Operations Research).*

## Sets and Parameters

- $N$: number of families of items; $i \in \{1, \ldots, N\}$.

- $n_i$: number of items in family $i$; $j \in \{1, \ldots, n_i\}$.

- $b$: knapsack capacity (non-negative integer).

- $f_i$: setup cost of family $i$ (non-negative integer).

- $d_i$: setup capacity consumption of family $i$ (non-negative integer).

- $p_{ij}$: profit of item $j$ of family $i$ (non-negative integer).

- $w_{ij}$: capacity consumption (weight) of item $j$ of family $i$ (non-negative integer).

## Decision Variables

- $x_{ij} \in \{0,1\}$: equals $1$ if item $j$ of family $i$ is placed in the knapsack, $0$ otherwise.

- $y_i \in \{0,1\}$: equals $1$ if the knapsack is set up to accept items belonging to family $i$, $0$ otherwise.

## Objective

$$\begin{align}
\text{maximize} \quad & \sum_{i=1}^{N} \sum_{j=1}^{n_i} p_{ij}\, x_{ij} - \sum_{i=1}^{N} f_i\, y_i \tag{1}
\end{align}$$

## Constraints

$$\begin{align}
\text{subject to} \quad & \sum_{i=1}^{N} \sum_{j=1}^{n_i} w_{ij}\, x_{ij} + \sum_{i=1}^{N} d_i\, y_i \leq b \tag{2}\\
& x_{ij} \leq y_i \quad && \forall\, j = 1, \ldots, n_i, \quad \forall\, i = 1, \ldots, N \tag{3}\\
& x_{ij} \in \{0,1\} \quad && \forall\, j = 1, \ldots, n_i, \quad \forall\, i = 1, \ldots, N \tag{4}\\
& y_i \in \{0,1\} \quad && \forall\, i = 1, \ldots, N \tag{5}
\end{align}$$
