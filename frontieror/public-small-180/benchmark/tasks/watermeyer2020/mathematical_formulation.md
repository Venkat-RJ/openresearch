# Original Formulation: Resource-Constrained Project Scheduling Problem with Partially Renewable Resources and General Temporal Constraints (RCPSP/max-$\pi$)

*Source: A branch-and-bound procedure for the resource-constrained project scheduling problem with partially renewable resources and general temporal constraints, Kai Watermeyer and Jürgen Zimmermann, 2020.*

## Sets and Indices

$$\begin{align*}
V &= \{0, 1, \ldots, n, n+1\} && \text{Set of activities ($0$ and $n+1$ fictitious start/end activities)} \\
E &\subseteq V \times V && \text{Set of activity pairs with general temporal constraints} \\
\mathcal{R} && && \text{Set of partially renewable resources} \\
\Pi_k &\subseteq \{1, 2, \ldots, \bar{d}\} && \text{Subset of planning-horizon periods on which resource $k \in \mathcal{R}$ is defined}
\end{align*}$$

## Parameters

$$\begin{align*}
n && && \text{Number of real activities} \\
p_i &\in \mathbb{Z}_{\geq 0} && \text{Processing time of activity $i \in V$ (fictitious activities: $p_0 = p_{n+1} = 0$)} \\
r_{ik}^{d} &\in \mathbb{Z}_{\geq 0} && \text{Per-period demand of activity $i \in V$ for resource $k \in \mathcal{R}$ (fictitious: $0$)} \\
\delta_{ij} &\in \mathbb{Z} && \text{Time lag for pair $(i,j) \in E$ (negative lags act as maximum time lags)} \\
R_k && && \text{Capacity of partially renewable resource $k \in \mathcal{R}$} \\
\bar{d} && && \text{Prescribed maximum project duration}
\end{align*}$$

Maximum project duration is enforced by $(n+1, 0) \in E$ with $\delta_{n+1,0} = -\bar{d}$.

**Resource usage (normalized formulation, Böttcher et al. 1999).** For a start time $S_i$ of activity $i \in V$: $$\begin{align*}
r_{ik}^{u}(S_i) &:= \bigl|\, ]S_i,\; S_i + p_i] \cap \Pi_k \,\bigr|
  && \text{number of periods of $\Pi_k$ during which $i$ is in execution} \\
r_{ik}^{c}(S_i) &:= r_{ik}^{u}(S_i) \cdot r_{ik}^{d}
  && \text{consumption of resource $k$ by activity $i$} \\
r_{k}^{c}(S) &:= \sum_{i \in V} r_{ik}^{c}(S_i)
  && \text{total consumption of resource $k$ under schedule $S$}
\end{align*}$$

## Decision Variables

$$\begin{align*}
S_i \in \mathbb{Z}_{\geq 0} \quad (i \in V) && \text{Start time of activity $i$; a schedule is } S = (S_i)_{i \in V}.
\end{align*}$$

## Objective and Constraints

The problem RCPSP/max-$\pi$ is stated as problem (P1): $$\begin{align*}
\text{Minimize} \quad & f(S) = S_{n+1} \\
\text{subject to} \quad & S_j - S_i \geq \delta_{ij} && ((i,j) \in E) \\
& \sum_{i \in V} r_{ik}^{c}(S_i) \leq R_k && (k \in \mathcal{R}) \\
& S_0 = 0 \\
& S_i \in \mathbb{Z}_{\geq 0} && (i \in V).
\end{align*}$$ (P1)
