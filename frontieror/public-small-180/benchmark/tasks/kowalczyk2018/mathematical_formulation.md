# Original Formulation: Parallel Machine Scheduling with Weighted Completion Times ($Pm\,||\,\sum w_j C_j$, WCT)

*Source: A Branch-and-Price Algorithm for Parallel Machine Scheduling Using ZDDs and Generic Branching, Daniel Kowalczyk, Roel Leus, 2018 (INFORMS Journal on Computing 30(4):768–782).*

## Sets and Indices

$$\begin{align*}
J &= \{1,\dots,n\} && \text{set of $n$ independent jobs, index $j$}\\
M &= \{1,\dots,m\} && \text{set of $m$ identical parallel machines}\\
\mathscr{F} &\subseteq 2^{J} && \text{set of all subsets $s\subset J$ that lead to a \emph{feasible} single-machine schedule}
\end{align*}$$ Jobs are indexed in nonincreasing order of the ratio $w_j/p_j$, i.e. $w_1/p_1 \ge w_2/p_2 \ge \cdots \ge w_n/p_n$ (Smith’s SWPT rule, Property 1); the unique schedule of a subset $s$ sequences its jobs in this order.

## Parameters

$$\begin{align*}
p_j \in \mathbb{N}_0 &&& \text{processing time of job $j\in J$}\\
w_j \in \mathbb{N}_0 &&& \text{weight (priority) of job $j\in J$}\\
p_{[i]} &&& \text{$i$-th smallest processing time among jobs in $J$; } p_{\max}=p_{[n]}\\
H_{\max} &= \sum_{j\in J}\tfrac{p_j}{m} + \tfrac{(m-1)}{m}\,p_{\max} && \text{(Property~3) max.\ latest completion time on any machine}\\
H_{\min} &= \tfrac{1}{m}\!\left(\sum_{j\in J}p_j - \sum_{h=1}^{m-1}p_{[n-h+1]}\right) && \text{(Property~4) min.\ latest completion time on any machine}\\
r_j,\ \bar{d}_j &&& \text{release date and deadline of job $j$ (time window)}
\end{align*}$$ For each $s\in\mathscr{F}$ and $j\in s$, the completion time of $j$ in $s$ is $C_j(s)=\sum_{k\in s:\,k\le j} p_k$, and the cost of schedule $s$ is $$\begin{equation}
c_s = \sum_{j\in s} w_j\!\left(\sum_{k\in s:\,k\le j} p_k\right). \tag{5}
\end{equation}$$ A subset $s\subset J$ belongs to $\mathscr{F}$ iff $r_j + p_j \le C_j(s) \le \bar{d}_j$ for all $j\in s$ and the completion time of the last job lies in $[H_{\min},H_{\max}]$.

## Decision Variables

$$\begin{align*}
\lambda_s \in \{0,1\}, \quad s\in\mathscr{F} &&& \lambda_s = 1 \text{ iff schedule } s \text{ is assigned to one machine}
\end{align*}$$

## Objective

$$\begin{equation}
\text{minimize}\quad \sum_{s\in\mathscr{F}} c_s\,\lambda_s \tag{6a}
\end{equation}$$

## Constraints

$$\begin{align}
\sum_{s\in\mathscr{F}:\,j\in s} \lambda_s &\ge 1 && \forall\, j\in J \tag{6b}\\
\sum_{s\in\mathscr{F}} \lambda_s &\le m \tag{6c}\\
\lambda_s &\in \{0,1\} && \forall\, s\in\mathscr{F} \tag{6d}
\end{align}$$ Constraints (6b) are the $n$ covering (assignment) constraints ensuring every job is assigned to at least one machine; (6c) is the single capacity constraint limiting the solution to $m$ machines; (6d) states the binary nature of the variables.
