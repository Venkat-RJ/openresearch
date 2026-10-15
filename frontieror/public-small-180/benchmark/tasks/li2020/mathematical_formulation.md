# Original Formulation: Bi-Level Partial Interdiction on Capacitated Hierarchical Facilities

*Source*: Forghani, Dehghanian, Salari, Ghiami (2020), “A bi-level model and solution methods for partial interdiction problem on capacitated hierarchical facilities,” *Computers & Operations Research* 114:104831. Equations (1)–(12). Original model is a bilevel min–max; we transcribe it in bilevel form (no KKT / LP-dual substitution).

## Sets and Indices

- $I$: set of customers, index $i$.

- $S_1$: existing facilities at level I (lower-level facilities).

- $S_2$: existing facilities at level II (upper-level facilities), indices $j,f$.

- $K = \{0,1,\ldots,k_{\max}\}$: interdiction intensity levels ($k=0$ means no interdiction).

## Parameters

- $\alpha,\beta,\gamma$: unit per-distance transportation costs (customer$\to$level I, customer$\to$level II, level I$\to$level II).

- $\alpha',\beta',\gamma'$: unit outsourcing costs (type I, type II, referential type II).

- $h^1_k,\,h^2_k$: cost of attacking a level I / level II facility at interdiction intensity $k$.

- $z_i$: demand at customer $i$; $\theta$: fraction receiving type I service; $\sigma$: fraction of type I recipients requiring referential type II.

- $B$: total interdiction budget.

- $c^1_j,\,c^2_j$: initial type I / type II capacity of facility $j$ ($c^2_j=0$ for $j \in S_1$).

- $d^1_k,\,d^2_k$: cumulative capacity-reduction ratio at interdiction level $k$ for level I / level II facilities.

- $\ell_{ij}$: distance from customer $i$ to facility $j$; $\ell'_{jf}$: distance between facilities $j$ and $f$.

## Decision Variables

#### Upper-level (leader, interdictor).

- $x_{jk} \in \{0,1\}$: $1$ if facility $j \in S_1 \cup S_2$ is interdicted at level $k \in K$.

#### Lower-level (follower, defender; all continuous $\ge 0$).

- $u^1_{ij}$: demand of customer $i$ allocated to facility $j \in S_1 \cup S_2$ for type I service.

- $u^2_{ij}$: demand of customer $i$ allocated to facility $j \in S_2$ for type II service.

- $u^3_{jf}$: demand referred from facility $j \in S_1 \cup S_2$ to facility $f \in S_2$ for referential type II service.

- $o^1_i,\,o^2_i$: outsourced type I / type II demand of customer $i$.

- $o^3_j$: outsourced referential type II demand originating at facility $j \in S_1 \cup S_2$.

## Bi-Level Formulation

#### Upper level (interdictor — leader).

$$\begin{align}
  \max_{x} \quad & H(x) \tag{1}\\[2pt]
  \text{s.t.} \quad
    & \sum_{k \in K} x_{jk} \;=\; 1,
      && \forall\, j \in S_1 \cup S_2, \tag{2}\\[2pt]
    & \sum_{j \in S_1} \sum_{k \in K} h^1_k\, x_{jk}
      + \sum_{j \in S_2} \sum_{k \in K} h^2_k\, x_{jk}
      \;\le\; B, \tag{3}\\[2pt]
    & x_{jk} \;\in\; \{0,1\},
      && \forall\, j \in S_1 \cup S_2,\; k \in K. \tag{4}
\end{align}$$

#### Lower level (defender — follower).

For fixed $x$, $H(x)$ is the optimal value of $$\begin{align}
  H(x) \;=\; \min_{u,o} \quad
    &\ \sum_{i \in I} \sum_{j \in S_1} \ell_{ij}\,\alpha\, u^1_{ij}
    + \sum_{i \in I} \sum_{j \in S_2} \ell_{ij}\,\beta\,  u^1_{ij}
    + \sum_{i \in I} \sum_{j \in S_2} \ell_{ij}\,\beta\,  u^2_{ij} \notag\\
    &\ + \sum_{j \in S_1 \cup S_2} \sum_{f \in S_2} \ell'_{jf}\,\gamma\, u^3_{jf}
    + \sum_{i \in I} \alpha'\, o^1_i
    + \sum_{i \in I} \beta'\,  o^2_i \notag\\
    &\ + \sum_{j \in S_1 \cup S_2} \gamma'\, o^3_j
    + \sigma \sum_{i \in I} \gamma'\, o^1_i
    \tag{5}\\[2pt]
  \text{s.t.} \quad
    & \theta\, z_i
      \;=\; \sum_{j \in S_1 \cup S_2} u^1_{ij} + o^1_i,
      && \forall\, i \in I, \tag{6}\\[2pt]
    & (1-\theta)\, z_i
      \;=\; \sum_{j \in S_2} u^2_{ij} + o^2_i,
      && \forall\, i \in I, \tag{7}\\[2pt]
    & \sigma \sum_{i \in I} u^1_{ij}
      \;=\; \sum_{f \in S_2} u^3_{jf} + o^3_j,
      && \forall\, j \in S_1 \cup S_2, \tag{8}\\[2pt]
    & \sum_{i \in I} u^1_{ij}
      \;\le\; c^1_j \;-\; c^1_j \sum_{k=1}^{k_{\max}} d^1_k\, x_{jk},
      && \forall\, j \in S_1, \tag{9}\\[2pt]
    & \sum_{i \in I} u^1_{ij}
      \;\le\; c^1_j \;-\; c^1_j \sum_{k=1}^{k_{\max}} d^2_k\, x_{jk},
      && \forall\, j \in S_2, \tag{10}\\[2pt]
    & \sum_{i \in I} u^2_{ij}
      + \sum_{f \in S_1 \cup S_2} u^3_{fj}
      \;\le\; c^2_j \;-\; c^2_j \sum_{k=1}^{k_{\max}} d^2_k\, x_{jk},
      && \forall\, j \in S_2, \tag{11}\\[2pt]
    & u^1_{ij},\, u^2_{ij},\, u^3_{jf},\, o^1_i,\, o^2_i,\, o^3_j
      \;\ge\; 0. \tag{12}
\end{align}$$

**Bilevel structure.** The lower-level variables satisfy $$(u^1,u^2,u^3,o^1,o^2,o^3) \;\in\;
    \arg\min_{u,o \ge 0}\
    \bigl\{\; (5) \;:\; \text{(6)--(11) hold at given } x \;\bigr\},$$ rendering (1)–(12) a bi-level (max–min) interdiction model.
