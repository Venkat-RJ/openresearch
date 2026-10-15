# Original Formulation: Two-Echelon Capacitated Vehicle Routing Problem with Grouping Constraints (2E-CVRPGC)

*Source: A branch-and-cut algorithm for the two-echelon capacitated vehicle routing problem with grouping constraints, Tian Liu, Zhixing Luo, Hu Qin, Andrew Lim, European Journal of Operational Research (2018).*

## Sets and Indices

- $G=(V,A)$: undirected graph on which the 2E-CVRPGC is defined.

- $V = V_0 \cup V_S \cup V_C$: node set, where $V_0=\{0\}$ is the depot, $V_S$ is the set of satellites, and $V_C$ is the set of customers.

- $A = A_1 \cup A_2$: arc set, where $A_1 = \{(i,j)\mid i,j\in V_0\cup V_S\}$ are the arcs connecting the depot and the satellites, and $A_2 = \{(i,j)\mid i,j\in V_S\cup V_C,\, i\neq j\}\setminus
          \{(i,j)\mid i,j\in V_S,\, i\neq j\}$ are the arcs connecting the satellites and the customers.

- $V_C = C_1 \cup \cdots \cup C_m$ with $C_i\cap C_j=\emptyset$ for all $i\neq j$: the $m$ disjoint customer groups; $\mathbb{C}=\{C_1,\ldots,C_m\}$ is the collection of all groups.

- $K_1$: set of available vehicles in the first echelon.

## Parameters

- $d_i$: demand of customer $i\in V_C$.

- $c_{i,j}$: travel cost of vehicles on arc $(i,j)\in A$.

- $Q_1$: capacity of vehicles in the first echelon.

- $Q_2$: capacity of vehicles in the second echelon.

- $Q_s$: capacity (throughput) of satellite $s\in V_S$.

- $h_s$: handling cost per unit of freight at satellite $s\in V_S$.

- $T_1$: number of available vehicles in the first echelon.

- $T_2$: number of available vehicles in the second echelon.

- $T_s$: maximum number of vehicles associated with satellite $s\in V_S$.

- $|V_S|$: number of satellites (used as the big-$M$ in the MTZ constraints).

## Decision Variables

- $x_{i,j,k}\in\{0,1\}$, $(i,j)\in A_1$, $k\in K_1$: equal to $1$ if arc $(i,j)$ is traveled by first-echelon vehicle $k$, and $0$ otherwise.

- $w_{s,k}\geq 0$, $s\in V_S$, $k\in K_1$: quantity delivered to satellite $s$ by first-echelon vehicle $k$.

- $u_{s,k}\geq 0$, $s\in V_S$, $k\in K_1$: position of satellite $s$ in the route of first-echelon vehicle $k$.

- $f_{i,j}\geq 0$, $(i,j)\in A_2$: load of the vehicle when it travels through arc $(i,j)$.

- $y_{i,j}\in\{0,1\}$, $(i,j)\in A_2$: equal to $1$ if a second-echelon vehicle travels through arc $(i,j)$, and $0$ otherwise.

- $z_{i,s}\in\{0,1\}$, $i\in V_C$, $s\in V_S$: equal to $1$ if customer $i$ is served by a vehicle from satellite $s$, and $0$ otherwise.

- $v_{C_h,s}\in\{0,1\}$, $h=1,\ldots,m$, $s\in V_S$: equal to $1$ if customer group $C_h\in\mathbb{C}$ is assigned to satellite $s$, and $0$ otherwise.

## Objective

$$\begin{align}
\min\ \sum_{k\in K_1}\sum_{(i,j)\in A_1} c_{i,j}\,x_{i,j,k}
   + \sum_{s\in V_S} h_s \sum_{k\in K_1} w_{s,k}
   + \sum_{(i,j)\in A_2} c_{i,j}\,y_{i,j} \tag{1}
\end{align}$$

## Constraints

$$\begin{align}
\sum_{(i,j)\in A_1} x_{i,j,k} &= \sum_{(j,i)\in A_1} x_{j,i,k},
   && \forall\, i\in V_0\cup V_S,\; k\in K_1 \tag{2}\\[2pt]
\sum_{(i,j)\in A_1} x_{i,j,k} &\leq 1,
   && \forall\, i\in V_S,\; k\in K_1 \tag{3}\\[2pt]
u_{i,k} + 1 &\leq u_{j,k} + |V_S|\,(1 - x_{i,j,k}),
   && \forall\, (i,j)\in A_1,\; k\in K_1 \tag{4}\\[2pt]
w_{s,k} &\leq Q_s \sum_{(s,i)\in A_1} x_{s,i,k},
   && \forall\, s\in V_S,\; k\in K_1 \tag{5}\\[2pt]
\sum_{s\in V_S} w_{s,k} &\leq Q_1,
   && \forall\, k\in K_1 \tag{6}\\[2pt]
\sum_{k\in K_1} w_{s,k} &\leq Q_s,
   && \forall\, s\in V_S \tag{7}\\[2pt]
\sum_{(i,j)\in A_2} y_{i,j} &= \sum_{(j,i)\in A_2} y_{j,i} = 1,
   && \forall\, i\in V_C \tag{8}\\[2pt]
\sum_{(s,i)\in A_2} y_{s,i} &\leq T_s,
   && \forall\, s\in V_S \tag{9}\\[2pt]
\sum_{s\in V_S}\sum_{(s,i)\in A_2} y_{s,i} &\leq T_2, && \tag{10}\\[2pt]
\sum_{(j,i)\in A_2} f_{j,i} &= \sum_{(i,j)\in A_2} f_{i,j} + d_i,
   && \forall\, i\in V_C \tag{11}\\[2pt]
d_j\,y_{i,j} \leq f_{i,j} &\leq (Q_2 - d_i)\,y_{i,j},
   && \forall\, (i,j)\in A_2 \tag{12}\\[2pt]
z_{j,s} &\geq z_{i,s} + y_{i,j} - 1,
   && \forall\, (i,j)\in A_2 \tag{13}\\[2pt]
z_{s,s} &= 1,
   && \forall\, s\in V_S \tag{14}\\[2pt]
z_{i,s} &= v_{C_h,s},
   && \forall\, i\in C_h,\; C_h\in\mathbb{C},\; s\in V_S \tag{15}\\[2pt]
\sum_{k\in K_1} w_{s,k} &= \sum_{(s,i)\in A_2} f_{s,i},
   && \forall\, s\in V_S \tag{16}\\[2pt]
x_{i,j,k} &\in \{0,1\},
   && \forall\, (i,j)\in A_1,\; k\in K_1 \tag{17}\\[2pt]
w_{s,k} &\geq 0,
   && \forall\, s\in V_S,\; k\in K_1 \tag{18}\\[2pt]
u_{s,k} &\geq 0,
   && \forall\, s\in V_S,\; k\in K_1 \tag{19}\\[2pt]
f_{i,j} &\geq 0,
   && \forall\, (i,j)\in A_2 \tag{20}\\[2pt]
y_{i,j} &\in \{0,1\},
   && \forall\, (i,j)\in A_2 \tag{21}\\[2pt]
z_{i,s} &\in \{0,1\},
   && \forall\, i\in V_C,\; s\in V_S \tag{22}\\[2pt]
v_{C_h,s} &\in \{0,1\},
   && \forall\, C_h\in\mathbb{C},\; s\in V_S \tag{23}
\end{align}$$
