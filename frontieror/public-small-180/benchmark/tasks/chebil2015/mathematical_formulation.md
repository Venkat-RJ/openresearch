# Original Formulation: Knapsack Problem with Setup (KPS)

*Source: A dynamic programming algorithm for the Knapsack Problem with Setup, Khalil Chebil, Mahdi Khemakhem, Computers & Operations Research, 2015.*

## Sets and Parameters

- $N$ : number of classes (families) of items, indexed by $i \in \{1, \ldots, N\}$.

- $n_i$ : number of items in class $i$, indexed by $j \in \{1, \ldots, n_i\}$.

- $b \in \mathbb{N}$ : knapsack capacity (non-negative integer).

- $f_i$ : setup cost of class $i$, a negative integer.

- $d_i$ : setup capacity consumption of class $i$, a non-negative integer.

- $c_{ij} \in \mathbb{N}^{N \times n_i}$ : profit of item $j$ of class $i$.

- $a_{ij} \in \mathbb{N}^{N \times n_i}$ : capacity consumption (weight) of item $j$ of class $i$.

## Decision Variables

- $x_{ij} \in \{0,1\}$ : equals $1$ if item $j$ of class $i$ is placed in the knapsack, $0$ otherwise.

- $y_i \in \{0,1\}$ : equals $1$ if the knapsack is set up to accept items belonging to class $i$, $0$ otherwise.

## Objective

$$\begin{align}
\text{Max } z &= \sum_{i=1}^{N} \sum_{j=1}^{n_i} c_{ij}\, x_{ij} + \sum_{i=1}^{N} f_i\, y_i \tag{1}
\end{align}$$

## Constraints

$$\begin{align}
\text{s.t.} \quad
& \sum_{i=1}^{N} \sum_{j=1}^{n_i} a_{ij}\, x_{ij} + \sum_{i=1}^{N} d_i\, y_i \leq b \tag{2} \\
& x_{ij} \leq y_i && \forall i \in \{1, \ldots, N\},\ \forall j \in \{1, \ldots, n_i\} \tag{3} \\
& x_{ij},\, y_i \in \{0,1\} && \forall i \in \{1, \ldots, N\},\ \forall j \in \{1, \ldots, n_i\} \tag{4}
\end{align}$$
