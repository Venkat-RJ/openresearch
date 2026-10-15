# Original Formulation: Multiple Depot Vehicle Scheduling Problem (MDVSP)

*Source: A Branch-and-Cut Algorithm for the Multiple Depot Vehicle Scheduling Problem, Ahmed Hadjar, Odile Marcotte, and François Soumis, Operations Research 54(1), pp. 130–149, 2006.*

## Sets and Indices

- $\{T_1, T_2, \ldots, T_n\}$ : set of trips (tasks); trip $T_i$ starts at time $a_i$ and ends at time $b_i$.

- $K = \{1, 2, \ldots, |K|\}$ : set of depot indices; depots are $D_1, D_2, \ldots, D_{|K|}$.

- $G = (V, A)$ : directed multigraph. The vertex set $V$ is the union of the trips $\{T_1,\ldots,T_n\}$ and the depots $\{D_1,\ldots,D_{|K|}\}$. Depot $D_k$ is assigned the vertex index $n+k$.

- The arc set $A$ contains all arcs of the form $(D_k, T_i)$ or $(T_i, D_k)$ (for $k = 1,\ldots,|K|$ and $i \le n$), and $|K|$ copies (one per colour $k$) of each compatible pair $(T_i, T_j)$. Each arc is denoted by a triple $(i,j,k)$ with tail $i$, head $j$, and colour $k$. Depots are treated as “colours.”

- A pair $(T_i, T_j)$ is *compatible* if $b_i + t_{ij} \le a_j$.

- $\delta^+(i) = \{\, j \mid (i,j,k) \in A \text{ for some } k \,\}$ : successors of $i$.

- $\delta^-(i) = \{\, j \mid (j,i,k) \in A \text{ for some } k \,\}$ : predecessors of $i$.

## Parameters

- $a_i$ : start time of trip $T_i$.

- $b_i$ : end time of trip $T_i$.

- $t_{ij}$ : travel time between the end point of $T_i$ and the starting point of $T_j$.

- $c_{ij}$ : cost incurred if a vehicle performs trip $T_i$ immediately before trip $T_j$ (independent of the depot where the vehicle is housed).

- $c_{n+k,\,j}$ : cost incurred if $T_j$ is the first trip undertaken by a vehicle housed at depot $D_k$.

- $c_{i,\,n+k}$ : cost incurred if $T_i$ is the last trip undertaken by a vehicle housed at depot $D_k$.

- $v_k$ : maximum number of vehicles that depot $D_k$ can house, $k \in K$.

## Decision Variables

- $X^k_{ij} \in \{0,1\}$ for each arc $(i,j,k) \in A$ : equals $1$ if and only if arc $(i,j,k)$ belongs to one of the routes (cycles) in a feasible assignment.

## Objective

The integer program is referred to in the paper as $(P)$: $$\begin{equation}
  \min \sum_{(i,j,k) \in A} c_{ij}\, X^k_{ij} \tag{1}
\end{equation}$$

## Constraints

$$\begin{align}
  & \sum_{k \in K} \sum_{j \in \delta^+(i)} X^k_{ij} = 1,
    && i = 1, 2, \ldots, n, \tag{2} \\[4pt]
  & \sum_{j=1}^{n} X^k_{n+k,\,j} \le v_k,
    && k \in K, \tag{3} \\[4pt]
  & \sum_{j \in \delta^-(i)} X^k_{ji} - \sum_{j \in \delta^+(i)} X^k_{ij} = 0,
    && k \in K,\ i = 1, 2, \ldots, n,\, n+k, \tag{4} \\[4pt]
  & X^k_{ij} \in \{0,1\}, && \text{for all } (i,j,k) \in A. \notag
\end{align}$$
