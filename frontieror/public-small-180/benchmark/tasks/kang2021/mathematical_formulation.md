# Original Formulation: Heterogeneous Drone-Truck Routing Problem (HDTRP)

*Source: An Exact Algorithm for Heterogeneous Drone-Truck Routing Problem, Munjeong Kang and Chungmok Lee, Transportation Science 55(5):1088–1112, 2021.*

## Sets and Indices

- $N$ : set of customers.

- $d$ : the (single) depot node, duplicated into two identical nodes $s$ and $t$.

- $s,\, t$ : duplicated depot nodes ($s$ = origin/start, $t$ = destination/end); both represent the physical depot $d$.

- $N_s := N \cup \{s\}$.

- $N_t := N \cup \{t\}$.

- $N_{st} := N \cup \{s, t\}$.

- $G(N_{st}, A)$ : directed graph.

- $A := \{(i,j) \mid i \in N_s,\, j \in N_t,\, i \neq j\}$ : set of arcs.

- $L$ : set of drones.

## Parameters

- $t^v_{ij}$ : truck travel time of arc $(i,j) \in A$ (includes the service time at node $j$).

- $B^l$ : battery capacity of drone $l \in L$ (how long the drone can be used for deliveries).

- $b^l_{ij}$ : total battery consumption required for drone $l \in L$ to complete the delivery operation from waiting node $i \in N_s$ for the demand of customer $j \in N$.

- $\tau^l_{ij}$ : total required time for drone $l \in L$ to complete the delivery from $i$ to $j$.

- $M$ : a sufficiently large number (big-$M$).

## Decision Variables

- $x_{ij} \in \{0,1\}$ : $1$ if the truck moves from $i$ to $j$, $0$ otherwise, $\forall (i,j) \in A$.

- $h^l_{ij} \in \{0,1\}$ : $1$ if drone $l \in L$ is dispatched at node $i$ for the delivery of customer $j$, $0$ otherwise, $\forall (i,j) \in A$.

- $v_i$ : continuous variable representing the visiting order of the truck at node $i$.

- $w_i$ : continuous variable representing the truck waiting time at node $i$.

## Objective

$$\begin{equation}
  \text{(P)} \quad \min \ \sum_{(i,j) \in A} t^v_{ij}\, x_{ij} \;+\; \sum_{i \in N_s} w_i \tag{1}
\end{equation}$$

## Constraints

$$\begin{align}
  & \sum_{j \in N} x_{sj} = 1, \tag{2}\\[2pt]
  & \sum_{i \in N} x_{it} = 1, \tag{3}\\[2pt]
  & \sum_{j \in N_t:\, j \neq i} x_{ij} = \sum_{j \in N_s:\, j \neq i} x_{ji}, && \forall\, i \in N, \tag{4}\\[2pt]
  & v_i - v_j \leq M(1 - x_{ij}) - 1, && \forall\, (i,j) \in A, \tag{5}\\[2pt]
  & \sum_{i \in N_s:\, i \neq j} x_{ij} + \sum_{i \in N_s:\, i \neq j} \sum_{l \in L} h^l_{ij} = 1, && \forall\, j \in N, \tag{6}\\[2pt]
  & M \sum_{j \in N_t:\, j \neq i} x_{ij} \geq \sum_{j \in N:\, j \neq i} \sum_{l \in L} h^l_{ij}, && \forall\, i \in N_s, \tag{7}\\[2pt]
  & \sum_{i \in N_s} \sum_{j \in N:\, j \neq i} b^l_{ij}\, h^l_{ij} \leq B^l, && \forall\, l \in L, \tag{8}\\[2pt]
  & w_i \geq \sum_{j \in N:\, j \neq i} \tau^l_{ij}\, h^l_{ij}, && \forall\, i \in N_s,\, l \in L, \tag{9}\\[2pt]
  & v_s = 0, \tag{10}\\[2pt]
  & x_{ij} \in \{0,1\}, && \forall\, (i,j) \in A, \tag{11}\\[2pt]
  & h^l_{ij} \in \{0,1\}, && \forall\, (i,j) \in A,\, l \in L. \tag{12}
\end{align}$$
