# Original Formulation: Two-Dimensional Vector Packing Problem (2DVPP)

*Source: A branch-and-price algorithm for the two-dimensional vector packing problem, Lijun Wei, Minghui Lai, Andrew Lim, Qian Hu, European Journal of Operational Research, 2019.*

The paper studies the 2DVPP and *first* states it (Section 2.1, “Set-partitioning formulation”) through the set-partitioning model $(F)$, eqs. (1)–(3). The variables $\lambda_p$ introduced here are carried into every later section (branching, master problem, pricing, experiments), so $(F)$ is the original formulation. The capacity data $(W,V,w_i,v_i)$ enters $(F)$ implicitly: it defines which patterns belong to the feasible pattern set $\mathscr{P}$ (see Remarks).

## Sets and Indices

- $I = \{1,2,\ldots,n\}$ : set of items, indexed by $i$.

- $\mathscr{P}$ : index set of all *feasible* packing patterns, indexed by $p$. A packing pattern is a set of items that can be loaded into one bin without violating any capacity constraint.

## Parameters

- $n$ : number of items.

- $W$ : weight limit (capacity) of each (identical) bin.

- $V$ : volume capacity of each (identical) bin.

- $w_i$ : weight of item $i$, $\ 0 < w_i \le W$, $\ \forall i \in I$.

- $v_i$ : volume of item $i$, $\ 0 < v_i \le V$, $\ \forall i \in I$.

- $a_{ip} \in \{0,1\}$ : equals $1$ if item $i$ is in pattern $p$ and $0$ otherwise. Pattern $p$ is the vector $\mathbf{a}_p = (a_{1p}, a_{2p}, \ldots, a_{np})^{T}$.

## Decision Variables

- $\lambda_p \in \{0,1\}$ : equals $1$ if and only if pattern $p$ is in the optimal solution and $0$ otherwise, $\ \forall p \in \mathscr{P}$.

## Objective

$$\begin{align}
(F) \quad z(F) = \min \quad & \sum_{p \in \mathscr{P}} \lambda_p \tag{1}
\end{align}$$

## Constraints

$$\begin{align}
\text{s.t.} \quad & \sum_{p \in \mathscr{P}} a_{ip}\,\lambda_p = 1, && \forall\, i \in I \tag{2} \\
                  & \lambda_p \in \{0,1\}, && \forall\, p \in \mathscr{P} \tag{3}
\end{align}$$
