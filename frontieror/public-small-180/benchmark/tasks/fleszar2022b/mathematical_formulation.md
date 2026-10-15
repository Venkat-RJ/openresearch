# Original Formulation: Quadratic Multiple Knapsack Problem (QMKP)

*Source: A branch-and-bound algorithm for the quadratic multiple knapsack problem, Krzysztof Fleszar, 2021/2022 (European Journal of Operational Research).*

## Sets and Indices

$$\begin{align*}
N &= \{1,\dots,n\} && \text{set of all items, indexed by } i,j \\
M &= \{1,\dots,m\} && \text{set of all knapsacks, indexed by } k
\end{align*}$$

## Parameters

$$\begin{align*}
w_i &: \text{positive integer weight of item } i \in N \\
v_i &: \text{individual value of item } i \in N \\
v_{ij} = v_{ji} &: \text{pairwise value of items } i,j \in N,\; i<j, \text{ earned only if both are packed in the same knapsack} \\
c_k &: \text{capacity of knapsack } k \in M
\end{align*}$$ The paper assumes all parameters are integers; weights are positive while individual and pairwise values may be positive, zero, or negative. Items are sorted by non-decreasing weight ($w_{i-1}\le w_i$) and all knapsacks have identical capacities ($c_k = c\;\forall k \in M$).

## Decision Variables

$$\begin{align*}
x_{ik} \in \{0,1\} \quad \forall i \in N,\; k \in M, \qquad
x_{ik} =
\begin{cases}
1 & \text{if item } i \text{ is packed in knapsack } k,\\
0 & \text{otherwise.}
\end{cases}
\end{align*}$$

## Objective

$$\begin{align}
\max \quad \sum_{k \in M} \left( \sum_{i \in N} v_i\, x_{ik} + \sum_{i \in N} \sum_{j \in N;\, i<j} v_{ij}\, x_{ik}\, x_{jk} \right) \tag{1}
\end{align}$$

## Constraints

$$\begin{align}
\sum_{i \in N} w_i\, x_{ik} &\leq c_k && \forall k \in M \tag{2}\\
\sum_{k \in M} x_{ik} &\leq 1 && \forall i \in N \tag{3}\\
x_{ik} &\in \{0,1\} && \forall i \in N,\; k \in M \tag{4}
\end{align}$$
