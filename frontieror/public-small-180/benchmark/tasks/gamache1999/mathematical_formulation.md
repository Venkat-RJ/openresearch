# Original Formulation: Generalized Set Partitioning Problem for Aircrew Rostering (GSPP)

*Source: A Column Generation Approach for Large-Scale Aircrew Rostering Problems, Gamache, Soumis, Marquis, Desrosiers, 1999 (Operations Research 47(2):247–263).*

## Sets and Parameters

- $p = 1, 2, \ldots, n$ index for (mother) pairings; $n$ is the number of pairings.

- $k = 1, 2, \ldots, m$ index for employees (crew members); $m$ is the number of employees.

- $R_k$ set of admissible rosters for employee $k$; $r \in R_k$ indexes a roster.

- $c_p$ cost coefficient corresponding to the duration of pairing $p$.

- $b_p$ number of times pairing $p$ must be covered (number of daughter pairings of $p$).

- $a_{pr}$ binary coefficient: $1$ if roster $r$ includes pairing $p$, $0$ otherwise.

- $t_r$ unproductive time on roster $r$ (time the employee is neither assigned to a pairing nor to a pre-assigned activity).

## Decision Variables

- $x_r \in \{0,1\}$ binary variable: $1$ if roster $r$ is selected, $0$ otherwise, for $r \in \bigcup_{k=1}^{m} R_k$.

- $e_p \geq 0$ slack variable associated with constraint set (2): number of uncovered copies of pairing $p$, for $p = 1, 2, \ldots, n$.

## Objective

$$\begin{equation}
\min \sum_{p=1}^{n} c_p e_p \tag{1}
\end{equation}$$

## Constraints

$$\begin{align}
\sum_{k=1}^{m} \sum_{r \in R_k} a_{pr}\, x_r + e_p &= b_p, & p &= 1, 2, \ldots, n, \tag{2}\\
\sum_{r \in R_k} x_r &= 1, & k &= 1, 2, \ldots, m, \tag{3}\\
e_p &\geq 0, & p &= 1, 2, \ldots, n, \tag{4}\\
x_r &\in \{0,1\}, & r &\in \bigcup_{k=1}^{m} R_k. \tag{5}
\end{align}$$
