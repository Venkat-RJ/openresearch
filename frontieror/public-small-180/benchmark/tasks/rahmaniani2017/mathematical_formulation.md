# Original Formulation: Uncapacitated Hub Location Problem with Multiple Assignments (UHLPMA) — Hamacher-style Ordered-Pair Formulation

**Source.** Contreras, Cordeau, and Laporte (2011), *Operations Research* 59(6), 1477–1490, Section 2, equations (1)–(5). The benchmark directory is named `rahmaniani2017` but the underlying formulation is Contreras et al. (2011), which follows Hamacher, Labbé, Nickel, Sonneborn (2004). This file uses the *ordered-pair* variables $x_{ijk}$ (commodity $k$ routed via ordered hub pair $(i,j)$) rather than the path/edge-based restatement.

## Sets and Indices

- $G = (N, A)$ — complete digraph on nodes $N$.

- $H \subseteq N$ — set of potential hub locations.

- $K$ — set of commodities (origin–destination pairs with positive flow).

- For each $k \in K$: $o(k)\in N$, $d(k)\in N$ are the origin and destination; $W_k \ge 0$ is the flow amount.

## Parameters

- $f_i$ — fixed set-up cost of opening a hub at $i \in H$.

- $d_{ij} \ge 0$ — transportation distance between $i$ and $j$.

- $\chi,\, \tau,\, \delta$ — collection, inter-hub transfer, and distribution cost parameters, with $0 < \tau < 1$ and $\tau < \chi,\ \tau < \delta$ (economies of scale between hubs).

- $\hat F_{ijk} \;=\; W_k\bigl(\chi\,d_{o(k)\,i} + \tau\,d_{ij} + \delta\,d_{j\,d(k)}\bigr)$ — unit transportation cost of routing commodity $k$ along the path $o(k)\!\to\!i\!\to\!j\!\to\!d(k)$ via the *ordered* hub pair $(i,j)$.

## Decision Variables

$$\begin{align*}
& z_i \in \{0,1\} && \forall\, i \in H, \quad z_i = 1 \text{ iff a hub is opened at $i$},\\
& x_{ijk} \ge 0  && \forall\, i,j \in H,\ k\in K, \quad \text{fraction of commodity $k$ routed via ordered hub pair $(i,j)$.}
\end{align*}$$

## Formulation

$$\begin{align}
\min\quad
& \sum_{i\in H} f_i\,z_i
  \;+\; \sum_{i\in H}\sum_{j\in H}\sum_{k\in K}
        \hat F_{ijk}\,x_{ijk} \tag{1}\\[2pt]
\text{s.t.}\quad
& \sum_{i\in H}\sum_{j\in H} x_{ijk} \;=\; 1
  && \forall\, k\in K, \tag{2}\\
& \sum_{j\in H} x_{ijk} \;+\; \sum_{j\in H\setminus\{i\}} x_{jik}
    \;\le\; z_i
  && \forall\, i\in H,\ \forall\, k\in K, \tag{3}\\
& x_{ijk} \;\ge\; 0
  && \forall\, i,j\in H,\ \forall\, k\in K, \tag{4}\\
& z_i \in \{0,1\}
  && \forall\, i \in H. \tag{5}
\end{align}$$

\(1\) minimizes the sum of hub opening costs and total transportation cost. (2) requires that each commodity $k$ is routed via exactly one ordered hub pair $(i,j)$. (3) allows flow through node $i$ (either as first hub, $x_{ijk}$, or second hub, $x_{jik}$) only if a hub is opened at $i$. (4)–(5) are the variable domains.
