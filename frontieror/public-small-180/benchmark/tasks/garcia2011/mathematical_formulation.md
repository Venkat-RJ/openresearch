# Sets and Parameters

- $n$: number of customers (nodes). Indices $i, j \in \{1, \ldots, n\}$.

- $p$: number of medians (facilities) to open, $1 \leq p \leq n-1$.

- $c_{ij} \geq 0$: allocation cost between customer $i$ and candidate median $j$, with $c_{ii}=0$ and $c_{ij}>0$ for $i \neq j$.

# Decision Variables

- $x_{jj} \in \{0,1\}$ for $1 \leq j \leq n$: location variable, $x_{jj}=1$ iff node $j$ is selected as a median.

- $x_{ij} \geq 0$ for $1 \leq i,j \leq n,\ i \neq j$: allocation variable, $x_{ij}=1$ iff customer $i$ is assigned to median $j$.

# Classical Formulation (CF)

$$\begin{align}
(\mathrm{CF})\quad \min \quad & \sum_{i=1}^{n} \sum_{j=1}^{n} c_{ij}\, x_{ij} \\
\text{s.t.}\quad & \sum_{j=1}^{n} x_{ij} = 1, && 1 \leq i \leq n, \tag{1} \\
& x_{ij} \leq x_{jj}, && 1 \leq i, j \leq n,\ i \neq j, \tag{2} \\
& \sum_{j=1}^{n} x_{jj} = p, \tag{3} \\
& x_{jj} \in \{0,1\}, && 1 \leq j \leq n, \\
& x_{ij} \geq 0, && 1 \leq i, j \leq n,\ i \neq j.
\end{align}$$

The objective minimizes the total allocation cost. Constraints (1) require every customer to be fully assigned. Constraints (2) forbid assignment to a node unless that node is chosen as a median. Constraint (3) fixes the number of open medians to $p$.
