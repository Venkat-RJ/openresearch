# Original Formulation: Vehicle Routing Problem with Time Windows and Convex Node Costs (VRPTW-CNC)

*Source: Branch-and-Cut-and-Price for the Vehicle Routing Problem with Time Windows and Convex Node Costs, Qie He, Stefan Irnich, Yongjia Song, 2019 (Transportation Science, Articles in Advance).*

## Sets and Parameters

$$\begin{align*}
N &= \{1, 2, \ldots, n\} && \text{set of customers,}\\
V &= N \cup \{0, n+1\} && \text{vertex set ($0$ and $n+1$ are the start/end depot copies),}\\
A && & \text{arc set of the directed graph $G = (V, A)$,}\\
\Omega && & \text{set of all feasible VRPTW routes,}\\
[i:j] &= \{i, i+1, \ldots, j\} && \text{integer interval for $i, j \in \mathbb{Z}$,}\\
K && & \text{number of homogeneous vehicles (fleet size),}\\
Q && & \text{vehicle capacity,}\\
q_i && & \text{demand of customer $i \in N$, with $q_0 = q_{n+1} = 0$,}\\
[e_i, \ell_i] && & \text{service time window of vertex $i \in V$ (given also for $0$ and $n+1$),}\\
\tau_{ij} && & \text{travel time on arc $(i,j) \in A$,}\\
s_i && & \text{service duration at vertex $i \in V$, with $s_0=s_{n+1}=0$,}\\
c_{ij} && & \text{routing cost on arc $(i,j) \in A$,}\\
f_i(t_i) && & \text{convex inconvenience cost of customer $i \in V$ as a function of}\\
        && & \text{the service start time $t_i$ (with $f_0 \equiv f_{n+1} \equiv 0$),}\\
a_{iP} && & \text{$=1$ if customer $i \in N$ is served by route $P \in \Omega$, $0$ otherwise.}
\end{align*}$$

A feasible route $P = (r_0, r_1, \ldots, r_m)$ is an elementary $0$–$(n+1)$ path in $G$ (i.e. $r_0 = 0$, $r_m = n+1$) of length $m > 1$. The route $P$ is *time-window feasible* if there exist schedule times $t_0, t_1, \ldots, t_m \in \mathbb{R}$ with $$\begin{align}
t_i &\in [e_{r_i}, \ell_{r_i}] && \forall i \in [0:m], \tag{1a}\\
t_{i-1} + s_{r_{i-1}} + \tau_{r_{i-1}, r_i} &\le t_i && \forall i \in [1:m]. \tag{1b}
\end{align}$$ The route $P$ respects vehicle capacity if $\sum_{i=0}^{m} q_{r_i} \le Q$. Its routing cost is $c_P = \sum_{i=1}^{m} c_{r_{i-1}, r_i}$, and its minimum inconvenience cost is the optimal value of the *service scheduling problem* $$\begin{equation}
f_P = \min\left\{ \sum_{i=0}^{m} f_{r_i}(t_i) \;\middle|\; t_0, t_1, \ldots, t_m \text{ satisfy } (1) \right\}. \tag{2}
\end{equation}$$ The total cost of route $P$ is $c_P + f_P$.

## Decision Variables

$$\begin{align*}
\lambda_P \in \mathbb{Z}_{\ge 0} && \text{number of times route $P \in \Omega$ is used.}
\end{align*}$$

## Objective

$$\begin{equation}
\min \quad \sum_{P \in \Omega} (c_P + f_P)\,\lambda_P \tag{3a}
\end{equation}$$

## Constraints

$$\begin{align}
\sum_{P \in \Omega} a_{iP}\,\lambda_P &= 1 && \forall i \in N, \tag{3b}\\
\sum_{P \in \Omega} \lambda_P &\le K, \tag{3c}\\
\lambda_P &\ge 0 \text{ integer} && \forall P \in \Omega. \tag{3d}
\end{align}$$
