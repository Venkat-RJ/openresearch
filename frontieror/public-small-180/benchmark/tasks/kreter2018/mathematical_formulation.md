# Original Formulation: Resource Availability Cost Problem (RACP)

*Source: Mixed-Integer Linear Programming and Constraint Programming Formulations for Solving Resource Availability Cost Problems, Stefan Kreter, Andreas Schutt, Peter J. Stuckey, Jürgen Zimmermann, 2018 (European Journal of Operational Research).*

## Sets and Indices

$$\begin{align*}
  & V := \{0, 1, \ldots, n, n+1\}     && \text{set of activities } (n \geq 1) \\
  & V^f \subset V                      && \text{fictitious activities } (p_i = 0,\ r_{ik} = 0\ \forall k);\ 0, n+1 \in V^f \\
  & V^r := V \setminus V^f             && \text{real activities (positive processing time)} \\
  & \mathcal{R}                        && \text{set of renewable resource types} \\
  & A                                  && \text{arc set; } \langle i,j \rangle \in A \text{ is a precedence relation} \\
  & N = (V, A)                         && \text{activity-on-node project network}
\end{align*}$$

## Parameters

$$\begin{align*}
  & n                                  && \text{number of non-dummy activities} \\
  & p_i \in \mathbb{N}_0               && \text{processing time of activity } i \in V \\
  & r_{ik} \in \mathbb{N}_0            && \text{per-period requirement of activity } i \text{ on resource } k \\
  & c_k                                && \text{cost for providing one unit of resource } k \in \mathcal{R} \\
  & \bar{d}                            && \text{prescribed upper bound on the project duration (must not be exceeded)}
\end{align*}$$

Derived quantities used in the formulation: $$\begin{align*}
  & C_i := S_i + p_i                                            && \text{completion time of activity } i \in V \text{ (non-interruptible)} \\
  & \mathcal{A}(S,t) := \{ i \in V^r \mid S_i \leq t < C_i \}   && \text{active set at time } t \in [0,\bar{d}] \text{ for schedule } S \\
  & r_k(S,t) := \sum_{i \in \mathcal{A}(S,t)} r_{ik}            && \text{utilization of resource } k \text{ at time } t
\end{align*}$$

## Decision Variables

$$\begin{align*}
  & S_i \in \mathbb{N}_0    && \text{start time of activity } i \in V,\quad \text{with } S_0 := 0 \text{ and } S_{n+1} \leq \bar{d} \\
  & R_k \geq 0              && \text{amount of resource } k \in \mathcal{R} \text{ to be provided (continuous)}
\end{align*}$$

## Objective

$$\begin{align}
  \text{Minimize} \quad & \sum_{k \in \mathcal{R}} c_k R_k                          \tag{1}
\end{align}$$

## Constraints

$$\begin{align}
  \text{subject to} \quad
    & S_j - S_i \geq p_i             & \langle i,j \rangle \in A           \tag{2} \\
    & r_k(S,t) \leq R_k              & k \in \mathcal{R},\ t \in [0,\bar{d}]  \tag{3} \\
    & S_i \in \{0,1,\ldots,\bar{d}\} & i \in V                            \tag{4} \\
    & R_k \geq 0                      & k \in \mathcal{R}                  \tag{5}
\end{align}$$
