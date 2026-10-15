# Original Formulation: Vehicle Routing with Demand Allocation Problem (VRDAP)

*Source: A Branch-and-Price Algorithm for a Vehicle Routing with Demand Allocation Problem, Mohammad Reihaneh and Ahmed Ghoniem, European Journal of Operational Research (2018/2019).*

The paper presents this 0–1 MIP as the definition of the VRDAP in Appendix A (Equations (4a)–(4l)), using the notation introduced in Section 3.1 (“Problem Statement”). Section 3.1 states explicitly that “the VRDAP can be modeled as a 0–1 MIP (as in the Appendix A) or equivalently as a set partitioning model.” The set-partitioning master program (1) is the column-generation reformulation used by the branch-and-price algorithm; the compact MIP below is the original problem-definition formulation (and the model solved directly by CPLEX/Gurobi in the computational study).

## Sets and Indices

- $S$: set of candidate delivery sites, indexed by $i$ (or $j$).

- $K$: set of customers, indexed by $k$.

- $V$: set of vehicle tours of capacity $Q$ available for delivery, indexed by $v$; $|V|$ is the maximum number of vehicles/tours.

- $N = S \cup \{0\}$: set of delivery sites augmented with node $0$, which represents the central depot.

- $E = \{(i,j) : i,j \in N,\ i \neq j\}$: set of routing arcs.

- $E' = \{(k,j) : k \in K,\ j \in S\}$: set of assignment arcs.

## Parameters

- $c_{ij}$: cost of routing arc $(i,j) \in E$.

- $f_{kj}$: cost of assigning customer $k$ to delivery site $j$, $(k,j) \in E'$ (written $f_{ik}$ in the objective with $i \in S$).

- $d_k$: demand of customer $k \in K$.

- $Q$: vehicle (tour) capacity.

## Decision Variables

- $s_{ik} \in \{0,1\}$: $s_{ik}=1 \Leftrightarrow$ customer $k$ is assigned to site $i$, $\forall i \in S,\ k \in K$.

- $e^{v}_{ij} \in \{0,1\}$: $e^{v}_{ij}=1 \Leftrightarrow$ arc $(i,j)$ is included in vehicle tour $v$, $\forall v \in V,\ (i,j) \in E$.

- $\theta^{v}_{ik} \in \{0,1\}$: $\theta^{v}_{ik}=1 \Leftrightarrow$ site $i$ is visited by vehicle tour $v$ and customer $k$ is assigned to site $i$, $\forall v \in V,\ i \in S,\ k \in K$.

- $q^{v}_{i} \ge 0$: total cumulative deliveries made upon serving site $i$ in vehicle tour $v$, $\forall v \in V,\ i \in S$.

## Objective

$$\begin{align}
\textbf{VRDAP}:\quad \min\ \sum_{v \in V} \sum_{(i,j) \in E} c_{ij}\, e^{v}_{ij}
   + \sum_{i \in S} \sum_{k \in K} f_{ik}\, s_{ik} \tag{4a}
\end{align}$$

## Constraints

$$\begin{align}
\sum_{v \in V} \sum_{i \in N \setminus \{j\}} e^{v}_{ij} &\le 1,
   & \forall j \in S \tag{4b}\\[2pt]
\sum_{j \in S} e^{v}_{0j} &\le 1,
   & \forall v \in V \tag{4c}\\[2pt]
\sum_{i \in N \setminus \{j\}} e^{v}_{ij} - \sum_{i \in N \setminus \{j\}} e^{v}_{ji} &= 0,
   & \forall v \in V,\ j \in N \tag{4d}\\[2pt]
q^{v}_{j} &\ge q^{v}_{i} + \sum_{k \in K} d_k\, s_{jk} - 2Q\bigl(1 - e^{v}_{ij}\bigr) + Q\, e^{v}_{ji},
   & \forall v \in V,\ i \neq j \in S \tag{4e}\\[2pt]
\theta^{v}_{ik} &\ge s_{ik} + \sum_{j \in N \setminus \{i\}} e^{v}_{ij} - 1,
   & \forall v \in V,\ i \in S,\ k \in K \tag{4f}\\[2pt]
\theta^{v}_{ik} &\le \sum_{j \in N \setminus \{i\}} e^{v}_{ij},
   & \forall v \in V,\ i \in S,\ k \in K \tag{4g}\\[2pt]
\theta^{v}_{ik} &\le s_{ik},
   & \forall v \in V,\ i \in S,\ k \in K \tag{4h}\\[2pt]
q^{v}_{i} &\le Q \sum_{j \in N \setminus \{i\}} e^{v}_{ij},
   & \forall v \in V,\ i \in S \tag{4i}\\[2pt]
\sum_{k \in K} d_k\, \theta^{v}_{ik} &\le q^{v}_{i},
   & \forall v \in V,\ i \in S \tag{4j}\\[2pt]
\sum_{v \in V} \sum_{i \in S} \theta^{v}_{ik} &= 1,
   & \forall k \in K \tag{4k}\\[2pt]
e,\, s \ \text{binary},\quad \theta,\, q &\ge 0. \tag{4l}
\end{align}$$
