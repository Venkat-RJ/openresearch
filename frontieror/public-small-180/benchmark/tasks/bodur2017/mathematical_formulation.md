# Original Formulation: Stochastic Capacitated Facility Location Problem (CAP) – Extensive Form

*Source: “Strengthened Benders Cuts for Stochastic Integer Programs with Continuous Recourse,” Bodur, Dash, Günlük, and Luedtke, INFORMS Journal on Computing 29(1):77–91, 2017. (Test problem of Section 4, equations (15a)–(15e).)*

## Sets and Parameters

- $\mathcal{I}$: set of potential facilities (indexed by $i$).

- $\mathcal{J}$: set of customers (indexed by $j$).

- $\mathcal{K}$: set of scenarios (indexed by $k$), each with probability $1/|\mathcal{K}|$.

- $f_i$: fixed opening cost of facility $i$.

- $s_i$: capacity of facility $i$.

- $q_{ij}$: per-unit flow cost from facility $i$ to customer $j$.

- $\lambda^k_j$: demand of customer $j$ under scenario $k$ (sampled from a normal distribution; see paper).

## Decision Variables

- $x_i \in \{0,1\}$: $1$ if facility $i$ is opened.

- $y^k_{ij} \ge 0$: amount of customer $j$’s demand satisfied by facility $i$ under scenario $k$.

## Objective

$$\begin{equation}
\min \sum_{i \in \mathcal{I}} f_i\, x_i \;+\; \frac{1}{|\mathcal{K}|} \sum_{k \in \mathcal{K}} \sum_{i \in \mathcal{I}} \sum_{j \in \mathcal{J}} q_{ij}\, y^k_{ij} \tag{15a}
\end{equation}$$

## Constraints

$$\begin{align}
\sum_{i \in \mathcal{I}} y^k_{ij} &\ge \lambda^k_j, & j \in \mathcal{J},\; k \in \mathcal{K} \tag{15b} \\
\sum_{j \in \mathcal{J}} y^k_{ij} &\le s_i\, x_i, & i \in \mathcal{I},\; k \in \mathcal{K} \tag{15c} \\
\sum_{i \in \mathcal{I}} s_i\, x_i &\ge \max_{k \in \mathcal{K}} \sum_{j \in \mathcal{J}} \lambda^k_j, & \tag{15d} \\
x \in \{0,1\}^{|\mathcal{I}|}, \quad y &\in \mathbb{R}_+^{|\mathcal{I}| \times |\mathcal{J}| \times |\mathcal{K}|}. & \tag{15e}
\end{align}$$
