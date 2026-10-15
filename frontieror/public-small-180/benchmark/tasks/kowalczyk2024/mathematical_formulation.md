# Original Formulation: Identical Parallel Machine Scheduling, Total Weighted Tardiness (TIF)

*Source: A flow-based formulation for parallel machine scheduling using decision diagrams, Daniel Kowalczyk, Roel Leus, Christopher Hojny, Stefan Røpke, 2024.*

The paper studies $Pm\,||\,\sum_{j} w_j T_j$ (identical parallel machine scheduling, minimize total weighted tardiness, no idle time). The first complete MILP that defines this problem in the paper’s own job-start variables is the *Time-Indexed Formulation* (TIF), Eq. (1) of Section 2.1. This is the canonical formulation reproduced below. The paper’s new flow/decision-diagram formulation (BDDF) and the arc-time-indexed formulation (ATIF) — alternative formulations of the *same* problem — are listed under **Variants**.

## Sets and Indices

$$\begin{align*}
J        &= \{1,\dots,n\}                     && \text{set of $n$ jobs} \\
M        &= \{1,\dots,m\}                     && \text{set of $m$ identical parallel machines} \\
j        &\in J                               && \text{job index} \\
t        &\in \{1,\dots,T\}                   && \text{period index; period $t$ starts at time $t-1$ and ends at $t$}
\end{align*}$$

## Parameters

$$\begin{align*}
p_j          &\in \mathbb{N}\setminus\{0\}   && \text{integer processing time of job $j\in J$} \\
d_j          &                                && \text{due date of job $j\in J$} \\
w_j          &                                && \text{weight (priority) of job $j\in J$} \\
p_{\max}     &= \max_{j\in J} p_j             && \text{maximum processing time} \\
T            &= \left\lceil \tfrac{\sum_{j\in J} p_j - p_{\max}}{m} \right\rceil + p_{\max}
                                              && \text{planning horizon (safe upper bound, no optimum lost)} \\
\widetilde{c}_{jt} &                          && \text{cost of job $j$ starting at the beginning of period $t$}
\end{align*}$$

## Decision Variables

$$\begin{align*}
y_{jt} \in \{0,1\} \quad && \text{$=1$ if job $j$ starts at the beginning of period $t$ (so $j$ ends in period $t+p_j-1$);} \\
                        && j\in J,\; t\in\{1,\dots,T-p_j+1\}.
\end{align*}$$

## Objective

$$\begin{align}
\min \quad \sum_{j\in J}\ \sum_{t=1}^{T-p_j+1} \widetilde{c}_{jt}\, y_{jt}
\tag{1a}
\end{align}$$

## Constraints

$$\begin{align}
\sum_{t=1}^{T-p_j+1} y_{jt} &= 1
&& \forall\, j\in J \tag{1b}\\[2pt]
\sum_{j\in J}\ \sum_{s=\max\{1,\,t-p_j+1\}}^{t} y_{js} &\le m
&& \forall\, t\in\{1,\dots,T\} \tag{1c}\\[2pt]
y_{jt} &\in \{0,1\}
&& \forall\, j\in J,\; t\in\{1,\dots,T\} \tag{1d}
\end{align}$$

Constraints (1b) ensure every job starts exactly once; Constraints (1c) impose that at most $m$ jobs are in process in any period (machine capacity).
