# Original Formulation: Resource-Constrained Project Scheduling Problem with Partially Renewable Resources and General Temporal Constraints (RCPSP/max-$\pi$)

*Source: A partition-based branch-and-bound algorithm for RCPSP/max-$\pi$, K. Watermeyer and J. Zimmermann, OR Spectrum 44:575–602, 2022.*

## Sets and Indices

$$\begin{align*}
  & V := \{0, 1, \ldots, n+1\} && \text{set of activities (index } i,j\text{)} \\
  & V' \subset V && \text{real activities } i \in V' \text{ with } p_i \in \mathbb{Z}_{>0} \\
  & V^e \subset V && \text{fictitious activities / events } i \in V^e \text{ with } p_i = 0 \\
  & && \text{(activity } 0 = \text{project start, } n+1 = \text{project end)} \\
  & E \subseteq V \times V && \text{activity pairs } (i,j) \text{ with start-to-start precedence} \\
  & \mathcal{R} && \text{set of partially renewable resources (index } k\text{)} \\
  & \Pi_k \subseteq \{1, 2, \ldots, \bar{d}\} && \text{subset of time periods assigned to resource } k \in \mathcal{R}
\end{align*}$$

## Parameters

$$\begin{align*}
  & n && \text{number of real activities} \\
  & p_i \in \mathbb{Z}_{>0} && \text{processing time of real activity } i \in V' \text{ (non-interruptible)} \\
  & p_i = 0 && \text{processing time of fictitious activity } i \in V^e \\
  & \delta_{ij} \in \mathbb{Z} && \text{time lag for } (i,j) \in E:\ \delta_{ij} \geq 0 \text{ min lag},\ \delta_{ij} < 0 \text{ max lag} \\
  & \bar{d} \in \mathbb{Z}_{\geq 0} && \text{maximum project duration (completion time)} \\
  & R_k \in \mathbb{Z}_{\geq 0} && \text{capacity of partially renewable resource } k \in \mathcal{R} \\
  & r^d_{ik} \in \mathbb{Z}_{\geq 0} && \text{per-period resource demand of activity } i \text{ for resource } k
\end{align*}$$

**Derived quantities** (functions of the start times, used inside the resource constraint): $$\begin{align*}
  r^u_{ik}(S_i) &:= \bigl|\, ]S_i,\, S_i + p_i] \cap \Pi_k \,\bigr|
    && \text{resource usage: \# periods of } \Pi_k \text{ in which } i \text{ executes} \\
  r^c_{ik}(S_i) &:= r^u_{ik}(S_i) \cdot r^d_{ik}
    && \text{resource consumption of activity } i \text{ for resource } k \\
  r^c_{k}(S) &:= \sum_{i \in V} r^c_{ik}(S_i)
    && \text{total consumption of resource } k \text{ by schedule } S
\end{align*}$$ Here $]S_i, S_i + p_i] = \{S_i + 1, S_i + 2, \ldots, S_i + p_i\}$, i.e. activity $i$ occupies period $t$ iff $S_i < t \leq S_i + p_i$.

## Decision Variables

$$\begin{align*}
  & S_i \in \mathbb{Z}_{\geq 0}, \quad i \in V && \text{start time of activity } i
\end{align*}$$ A schedule is the vector of start times $S = (S_i)_{i \in V}$.

## Objective

The RCPSP/max-$\pi$ is stated as problem (P): $$\begin{align}
  \text{Minimize} \quad & S_{n+1} \tag{P}
\end{align}$$

## Constraints

Subject to $S \in \mathcal{S}_T \cap \mathcal{S}_R$, where the time-feasible region is $$\begin{align}
  \mathcal{S}_T := \Bigl\{\, S \in \mathbb{Z}_{\geq 0}^{n+2} \ \Big|\ S_j - S_i \geq \delta_{ij} \ \text{ for all } (i,j) \in E \ \wedge\ S_0 = 0 \,\Bigr\}
  \tag{$\mathcal{S}_T$}
\end{align}$$ and the resource-feasible region is $$\begin{align}
  \mathcal{S}_R := \Bigl\{\, S \in \mathbb{Z}_{\geq 0}^{n+2} \ \Big|\ r^c_{k}(S) \leq R_k \ \text{ for all } k \in \mathcal{R} \,\Bigr\}.
  \tag{$\mathcal{S}_R$}
\end{align}$$ Equivalently, written out as the constraint system of (P): $$\begin{align}
  & S_j - S_i \geq \delta_{ij} && \forall\, (i,j) \in E \\
  & S_0 = 0 && \\
  & r^c_{k}(S) = \sum_{i \in V} \bigl|\, ]S_i,\, S_i + p_i] \cap \Pi_k \,\bigr| \cdot r^d_{ik} \ \leq\ R_k && \forall\, k \in \mathcal{R} \\
  & S_i \in \mathbb{Z}_{\geq 0} && \forall\, i \in V
\end{align}$$ The feasible region is $\mathcal{S} := \mathcal{S}_T \cap \mathcal{S}_R$; a schedule $S$ solving (P) is called optimal, and $\mathcal{OS}$ denotes the set of all optimal schedules.
