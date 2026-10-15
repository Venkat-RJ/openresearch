# Original Formulation: Service Network Design with a Heterogeneous Fleet (SND)

*Source: A hybrid algorithm for large-scale service network design considering a heterogeneous fleet, Z. Wang, M. Qi, C. Cheng, C. Zhang, European Journal of Operational Research 276 (2019) 483–494.*

## Sets and Indices

- $G=(N,A)$: directed graph with node set $N$ and arc set $A$; index $(i,j)\in A$.

- $K$: set of commodities; index $k\in K$. Commodity $k$ has demand $d^k$, origin node $o(k)$, and destination node $d(k)$.

- $F$: set of fleet (vehicle) types; index $f\in F$.

- $N_i^{+}=\{\,j\in N \mid (i,j)\in A\,\}$: set of successor nodes of node $i$.

- $N_i^{-}=\{\,j\in N \mid (j,i)\in A\,\}$: set of predecessor nodes of node $i$.

## Parameters

- $c_{ij}^{k}$: unit flow cost of transporting commodity $k\in K$ on arc $(i,j)\in A$.

- $h_{ij}^{f}$: fixed cost of utilizing one vehicle of fleet type $f\in F$ on arc $(i,j)\in A$.

- $u^{f}$: capacity of fleet type $f\in F$.

- $d^{k}$: demand of commodity $k\in K$.

- $w_i^{k}$: net supply/demand of commodity $k$ at node $i$, defined as $$w_i^{k}=
    \begin{cases}
      d^{k} & \text{if } i=o(k),\\
      -d^{k} & \text{if } i=d(k),\\
      0 & \text{otherwise.}
    \end{cases}$$

There is no limit on the number of vehicles of each type.

## Decision Variables

- $x_{ij}^{k}\ge 0$: continuous variable, the amount of flow of commodity $k\in K$ on arc $(i,j)\in A$.

- $y_{ij}^{f}\in\mathbb{Z}_0^{+}$: integer variable, the number of vehicles of fleet type $f\in F$ providing transportation services on arc $(i,j)\in A$.

## Objective

$$\begin{equation}
\min \ \sum_{k\in K}\sum_{(i,j)\in A} c_{ij}^{k} x_{ij}^{k}
   + \sum_{f\in F}\sum_{(i,j)\in A} h_{ij}^{f} y_{ij}^{f}
\tag{1}
\end{equation}$$

## Constraints

$$\begin{align}
\sum_{j\in N_i^{+}} x_{ij}^{k} - \sum_{j\in N_i^{-}} x_{ji}^{k} &= w_i^{k}
  && \forall\, i\in N,\ \forall\, k\in K, \tag{2}\\[4pt]
\sum_{j\in N_i^{+}} y_{ij}^{f} - \sum_{j\in N_i^{-}} y_{ji}^{f} &= 0
  && \forall\, i\in N,\ \forall\, f\in F, \tag{3}\\[4pt]
\sum_{k\in K} x_{ij}^{k} &\le \sum_{f\in F} u^{f} y_{ij}^{f}
  && \forall\, (i,j)\in A, \tag{4}\\[4pt]
x_{ij}^{k} &\ge 0
  && \forall\, (i,j)\in A,\ \forall\, k\in K, \tag{5}\\[4pt]
y_{ij}^{f} &\in \mathbb{Z}_0^{+}
  && \forall\, (i,j)\in A,\ \forall\, f\in F. \tag{6}
\end{align}$$
