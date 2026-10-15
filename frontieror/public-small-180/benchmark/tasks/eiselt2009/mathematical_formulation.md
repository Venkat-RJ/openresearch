# Sets and Parameters

- $N$: set of all nodes (origins, destinations, potential hub locations); indices $i, j \in N$.

- $\mathrm{Own}$: set of candidate hub nodes for the entering airline (“us”); indices $k, \ell \in \mathrm{Own}$.

- $\mathrm{Comp}$: set of hub nodes of existing (competing) airlines; $|\mathrm{Comp}| = r$.

- $s$: airline index; $s = o$ denotes the entering airline, $s = 1,2,\ldots$ denote incumbent airlines.

- $h_{ij}$: known non-elastic demand for air traffic from origin $i$ to destination $j$.

- $p$: number of hubs to be located by the entering airline (exogenously given).

- $A_{k\ell s}$: basic attractiveness of the hub pair $(k, \ell)$ operated by airline $s$.

- $\beta, \lambda$: attraction-decay exponents for travel time and travel cost, respectively.

- $\gamma \in [0,1]$: relative weight of travel time vs. travel cost.

- $\alpha \leq 1$: discount factor for interhub fare (economies of scale).

- $t_{ijk\ell s}$: total travel time for route $(i,j,k,\ell)$ by airline $s$.

- $c_{ijk\ell s}$: total fare for route $(i,j,k,\ell)$ by airline $s$.

# Derived Quantities

Utility (attraction) of route $(i,j,k,\ell)$ by airline $s$ (Equation (1)): $$\begin{equation}
u_{ijk\ell s} \;=\; \frac{A_{k\ell s}}{\gamma\, t_{ijk\ell s}^{\beta} \;+\; (1-\gamma)\, c_{ijk\ell s}^{\lambda}} \tag{1}
\end{equation}$$

Probability that a customer traveling from $i$ to $j$ uses route $(k,\ell)$ via the entering airline $o$ (Equation (2)): $$\begin{equation}
p_{ijk\ell o} \;=\; \frac{u_{ijk\ell o}}{\displaystyle \sum_{m, n \in \mathrm{Own}} u_{ijmno} \;+\; \sum_{s \neq o} \sum_{m, n \in \mathrm{Comp}} u_{ijmns}} \tag{2}
\end{equation}$$

# Decision Variables

$$\begin{align*}
y_k &\in \{0,1\} && \forall\, k \in \mathrm{Own} \quad (\text{1 if hub is located at $k$ by entering airline}) \\
w_{k\ell} &\in \{0,1\} && \forall\, k, \ell \in \mathrm{Own} \quad (\text{1 if hubs are located at \emph{both} $k$ and $\ell$})
\end{align*}$$

# Objective (nonlinear sum-of-ratios form, Equation (3))

$$\begin{align}
\mathrm{P}: \quad \max\; z
&= \sum_{i,j} \sum_{k, \ell \in \mathrm{Own}} h_{ij} \, p_{ijk\ell o} \notag \\
&= \sum_{i,j} h_{ij}\, \frac{\displaystyle\sum_{k, \ell \in \mathrm{Own}} w_{k\ell}\, u_{ijk\ell o}}{\displaystyle\sum_{m, n \in \mathrm{Own}} w_{mn}\, u_{ijmno} \;+\; \sum_{s \neq o} \sum_{m, n \in \mathrm{Comp}} u_{ijmns}} \tag{3}
\end{align}$$

# Constraints

$$\begin{align}
w_{k\ell} &\leq y_k && \forall\, k, \ell \in \mathrm{Own} \tag{4} \\
w_{k\ell} &\leq y_\ell && \forall\, k, \ell \in \mathrm{Own} \tag{5} \\
\sum_{k} y_k &= p \tag{6} \\
y_k,\; w_{k\ell} &\in \{0,1\} && \forall\, k, \ell \in \mathrm{Own} \tag{7}
\end{align}$$
