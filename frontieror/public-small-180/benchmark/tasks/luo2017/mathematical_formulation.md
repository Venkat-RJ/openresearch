# Original Formulation: Split-Delivery VRP with Time Windows and Linear Weight-Related Cost (SDVRPTWL) — Arc-Flow Model

*Source*: Luo, Qin, Zhu, Lim (2017), “Branch and Price and Cut for the Split-Delivery Vehicle Routing Problem with Time Windows and Linear Weight-Related Cost,” *Transportation Science*, Section 3 (arc-flow formulation). Equations (1)–(9).

## Sets and Indices

- $V = \{0,1,\ldots,n,n+1\}$: vertex set; vertex $0$ is the exit from the depot, vertex $n+1$ is the entrance to the depot.

- $V_C = \{1,\ldots,n\}$: set of $n$ customers.

- $E = \{(i,j) : i,j \in V,\ i \ne j,\ i \ne n+1,\ j \ne 0\}$: directed edge set.

- $V^{+}(i) = \{j \in V : e_i + s_i + t_{i,j} \le l_j,\ (i,j) \in E\}$: immediate successors of $i$.

- $V^{-}(i) = \{j \in V : e_j + s_j + t_{j,i} \le l_i,\ (j,i) \in E\}$: immediate predecessors of $i$.

- $K = \{1,2,\ldots,m\}$: set of $m$ available (homogeneous) vehicles.

## Parameters

- $d_i$: weight demand of customer $i \in V_C$ ($d_0 = +\infty$, $d_{n+1} = 0$).

- $s_i$: service time at vertex $i$ ($s_0 = s_{n+1} = 0$).

- $[e_i, l_i]$: time window at vertex $i$ ($e_0=e_{n+1}=0$, $l_0=l_{n+1}=+\infty$).

- $c_{i,j} \ge 0$: distance of edge $(i,j)$.

- $t_{i,j} \ge 0$: travel time of edge $(i,j)$ (distances and travel times satisfy the triangle inequality).

- $Q$: vehicle weight capacity (homogeneous fleet).

- $M$: a sufficiently large positive constant (big-$M$).

- $a,b$: parameters of the linear weight-related cost $f(w) = a w + b$.

## Decision Variables

- $x_{i,j,k} \in \{0,1\}$: $1$ if vehicle $k$ traverses edge $(i,j)$.

- $w_{i,j,k} \ge 0$: load weight of vehicle $k$ on edge $(i,j)$.

- $a_{i,k} \ge 0$: service-start time of vehicle $k$ at vertex $i$.

## Objective

$$\begin{equation}
  \min \ \sum_{k \in K}
         \sum_{i \in V_C \cup \{0\}}
         \sum_{j \in V^{+}(i)}
         c_{i,j}\,\bigl(a \cdot w_{i,j,k} + b \cdot x_{i,j,k}\bigr).
  \tag{1}
\end{equation}$$

## Constraints

$$\begin{align}
  \sum_{k \in K}
  \Bigl(\sum_{j \in V^{-}(i)} w_{j,i,k} \;-\; \sum_{j \in V^{+}(i)} w_{i,j,k}\Bigr)
     &\;\ge\; d_i,
    && \forall\, i \in V_C, \tag{2}\\[2pt]
  \sum_{k \in K} \sum_{j \in V^{-}(i)} x_{j,i,k}
     &\;\ge\; \lceil d_i / Q \rceil,
    && \forall\, i \in V_C, \tag{3}\\[2pt]
  \sum_{i \in V_C \cup \{n+1\}} x_{0,i,k}
     &\;=\; 1,
    && \forall\, k \in K, \tag{4}\\[2pt]
  \sum_{j \in V^{+}(i)} x_{i,j,k}
     \;=\; \sum_{j \in V^{-}(i)} x_{j,i,k}
     &\;\le\; 1,
    && \forall\, k \in K,\ i \in V_C, \tag{5}\\[2pt]
  \sum_{i \in V_C \cup \{0\}} x_{i,n+1,k}
     &\;=\; 1,
    && \forall\, k \in K, \tag{6}\\[2pt]
  w_{i,j,k} &\;\le\; Q\, x_{i,j,k},
    && \forall\, k \in K,\ i \in V_C \cup \{0\},\ j \in V^{+}(i), \tag{7}\\[2pt]
  a_{j,k} &\;\ge\; a_{i,k} + s_i + t_{i,j} + M\bigl(x_{i,j,k} - 1\bigr),
    && \forall\, k \in K,\ i \in V_C \cup \{0\},\ j \in V^{+}(i), \tag{8}\\[2pt]
  e_i \;\le\; a_{i,k} &\;\le\; l_i,
    && \forall\, k \in K,\ i \in V. \tag{9}
\end{align}$$

## Variable Domains

$$\begin{equation*}
  x_{i,j,k} \in \{0,1\},\qquad
  w_{i,j,k} \ge 0,\qquad
  a_{i,k} \ge 0,
  \quad \forall\, k \in K,\ i \in V_C \cup \{0\},\ j \in V^{+}(i).
\end{equation*}$$
