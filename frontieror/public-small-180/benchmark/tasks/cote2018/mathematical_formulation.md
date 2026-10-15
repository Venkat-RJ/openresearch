# Original Formulation: One-Dimensional Cutting Stock Problem (CSP)

*Source: The Meet-in-the-Middle Principle for Cutting and Packing Problems, Jean-François Côté and Manuel Iori, 2018.*

## Sets and Parameters

- $m$: number of item types; item type $i$ has integer width $w_i$ and integer demand $d_i$.

- $n = \sum_{i=1}^{m} d_i$: total number of item copies.

- $W$: integer capacity (width) of each identical bin.

- $G = (V,A)$: digraph with vertex set $V = \{0,1,\ldots,W\}$ (a vertex $q$ is a partial bin filling).

- $A$: set of arcs $(p,q)$, each representing either (i) the packing of an item of width $q-p$ starting at the partial filling $p$ (*item arc*), or (ii) an empty portion of the bin between fillings $p$ and $q$ (*loss arc*).

- $\delta^-(q)$: set of arcs entering vertex $q$; $\delta^+(q)$: set of arcs leaving vertex $q$.

## Decision Variables

- $x_{pq} \in \mathbb{Z}_{\ge 0}$: number of times arc $(p,q)\in A$ is selected.

- $z$: number of bins used, i.e. the amount of flow sent from vertex $0$ to vertex $W$ (each bin packing corresponds to a path from $0$ to $W$).

## Objective

$$\begin{align}
  \min \quad & z \tag{11}
\end{align}$$

## Constraints

$$\begin{align}
  \text{s.t.} \quad
  & \sum_{(q,r)\in\delta^+(q)} x_{qr} \;-\; \sum_{(p,q)\in\delta^-(q)} x_{pq}
    = \begin{cases}
        z   & \text{if } q = 0, \\
        -z  & \text{if } q = W, \\
        0   & \text{if } q = 1,2,\ldots,W-1,
      \end{cases} \tag{12} \\[4pt]
  & \sum_{(q,\,q+w_i)\in A} x_{q,\,q+w_i} \;\ge\; d_i, \quad i = 1,2,\ldots,m, \tag{13} \\[4pt]
  & x_{pq} \ge 0, \text{ integer}, \quad (p,q)\in A. \tag{14}
\end{align}$$

Constraints (12) impose flow conservation; constraints (13) state that all item demands must be fulfilled. Each possible packing of a bin is a path from $0$ to $W$, and the aim is to minimize the number $z$ of selected paths. The “$\ge$” in (13) (rather than “$=$”) allows over-production, which is admissible for the CSP.
