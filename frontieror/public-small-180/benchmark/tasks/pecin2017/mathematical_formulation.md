# Original Formulation: Vehicle Routing Problem with Time Windows (VRPTW)

*Source: New Enhancements for the Exact Solution of the Vehicle Routing Problem with Time Windows, Diego Pecin, Claudio Contardo, Guy Desaulniers, Eduardo Uchoa, 2017.*

## Sets and Indices

$$\begin{align*}
V        &= \{0, 1, \dots, n+1\} && \text{set of nodes; nodes $0$ and $n+1$ are two copies of the depot}\\
V^{+}    &= V \setminus \{0, n+1\} && \text{set of customers}\\
A        &= \{(i,j): i \in V\setminus\{n+1\},\ j \in V\setminus\{0\},\ i \ne j,\\
         &\qquad\ d_i + d_j \le Q,\ e_i + s_i + t_{ij} \le l_j\} && \text{set of feasible arcs}\\
\Omega   &&& \text{set of all feasible routes}
\end{align*}$$

## Parameters

$$\begin{align*}
d_i      &\quad \text{demand of customer } i \in V^{+},\ d_i > 0; \quad d_0 = d_{n+1} = 0\\
[e_i, l_i] &\quad \text{time window of node } i,\ 0 \le e_i \le l_i; \quad [e_0,l_0]=[e_{n+1},l_{n+1}]=[0,H]\\
s_i      &\quad \text{service time at customer } i \in V^{+},\ s_i > 0; \quad s_0 = s_{n+1} = 0\\
H        &\quad \text{planning horizon length},\ H > 0\\
c_{ij}   &\quad \text{traveling cost on arc } (i,j),\ c_{ij} \ge 0\\
t_{ij}   &\quad \text{traveling time on arc } (i,j),\ t_{ij} \ge 0\\
Q        &\quad \text{capacity of each (identical, unlimited fleet) vehicle},\ Q > 0\\
c_r      &\quad \text{cost of route } r \in \Omega:\quad
            c_r = c_{0,i_1} + \sum_{h=1}^{k-1} c_{i_h,i_{h+1}} + c_{i_k,n+1}\\
a^r_i    &\quad \text{number of visits to customer } i \in V^{+} \text{ in route } r \in \Omega
\end{align*}$$

*Route feasibility.* A route $r=(i_0=0,i_1,\dots,i_k,i_{k+1}=n+1)$ visiting $k\ge 1$ customers $i_1,\dots,i_k \in V^{+}$ is feasible if (i) $\sum_{h=1}^{k} d_{i_h} \le Q$ (capacity); (ii) $e_{i_h} \le T_h \le l_{i_h}$ for $1 \le h \le k$ (time windows); and (iii) $T_{k+1} \le H$ (route end time), where $T_0 = 0$, $T_h = \max\{e_{i_h},\, T_{h-1}+s_{i_{h-1}}+t_{i_{h-1},i_h}\}$ for $h>0$, and $T_{k+1} = T_k + s_{i_k} + t_{i_k,n+1}$.

## Decision Variables

$$\begin{align*}
\lambda_r \in \{0,1\} &\quad \text{equals 1 if route } r \in \Omega \text{ is selected in the solution, 0 otherwise}
\end{align*}$$

## Objective

$$\begin{align}
\min\ \sum_{r \in \Omega} c_r\, \lambda_r \tag{1}
\end{align}$$

## Constraints

$$\begin{align}
\sum_{r \in \Omega} a^r_i\, \lambda_r &= 1, & \forall i \in V^{+} \tag{2}\\
\lambda_r &\in \{0,1\}, & \forall r \in \Omega \tag{3}
\end{align}$$

The objective (1) minimizes the overall cost of the selected routes. Constraints (2) guarantee that each customer is served by exactly one route. Constraints (3) force the route-selection variables to be binary.
