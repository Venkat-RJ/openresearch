# Original Formulation: Steiner Tree-Star Problem (STS)

*Source*: Lee, Chiu, Ryan (1996), “A Branch and Cut Algorithm for a Steiner Tree-Star Problem,” *INFORMS Journal on Computing* 8(3):194–201. Equations (1)–(9) in the paper.

## Sets and Indices

- $M = \{1,\ldots,m\}$: set of target nodes; index $i \in M$.

- $N = \{1,\ldots,n\}$: set of Steiner nodes; indices $j,k \in N$ (with $j<k$ for edge variables between Steiner nodes).

## Parameters

- $b_j \ge 0$: fixed cost if Steiner node $j \in N$ is selected (active).

- $c_{ij} \ge 0$: connection cost of assigning target node $i \in M$ to Steiner node $j \in N$.

- $d_{jk} \ge 0$: connection cost between distinct Steiner nodes $j,k \in N$ (with $j<k$).

## Decision Variables

- $x_{ij} \in \{0,1\}$: $1$ if target node $i$ connects to Steiner node $j$; $0$ otherwise.

- $u_j \in \{0,1\}$: $1$ if Steiner node $j$ is selected (active); $0$ otherwise.

- $y_{jk} \ge 0$ (for $j<k$): $1$ if Steiner nodes $j$ and $k$ are joined by an edge of the Steiner spanning tree. Integrality is implied by the polyhedral structure given integrality of $u_j$.

## Objective

$$\begin{equation}
  \min \ \sum_{i \in M} \sum_{j \in N} c_{ij}\, x_{ij}
       + \sum_{j \in N} \sum_{\substack{k \in N\\k>j}} d_{jk}\, y_{jk}
       + \sum_{j \in N} b_j\, u_j.
  \tag{OBJ}
\end{equation}$$

## Constraints

$$\begin{align}
  \sum_{j \in N} x_{ij} &\;=\; 1,
    && \forall\, i \in M, \tag{1}\\[2pt]
  x_{ij} &\;\le\; u_j,
    && \forall\, i \in M,\; j \in N, \tag{2}\\[2pt]
  y_{jk} &\;\le\; u_j,
    && \forall\, j,k \in N,\; j<k, \tag{3}\\[2pt]
  y_{jk} &\;\le\; u_k,
    && \forall\, j,k \in N,\; j<k, \tag{4}\\[2pt]
  \sum_{j \in N} \sum_{\substack{k \in N\\k>j}} y_{jk}
     &\;=\; \sum_{j \in N} u_j \;-\; 1, \tag{5}\\[2pt]
  \sum_{j \in H} \sum_{\substack{k \in H\\k>j}} y_{jk}
     &\;\le\; \sum_{j \in H\setminus\{\ell\}} u_j,
    && \forall\, \ell \in H,\; H \subset N,\; |H| \ge 3, \tag{6}\\[2pt]
  x_{ij} &\;\ge\; 0,
    && \forall\, i \in M,\; j \in N, \tag{7}\\[2pt]
  y_{jk} &\;\ge\; 0,
    && \forall\, j,k \in N,\; j<k, \tag{8}\\[2pt]
  u_j &\;\in\; \{0,1\},
    && \forall\, j \in N. \tag{9}
\end{align}$$
