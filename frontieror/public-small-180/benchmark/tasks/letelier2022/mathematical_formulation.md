# Original Formulation: Bin Packing Problem with Time Lags (BPPTL)

*Source: Bin Packing Problem with Time Lags, Orlando Rivera Letelier, François Clautiaux, Ruslan Sadykov, 2022 (INFORMS Journal on Computing, 34(4), 2249–2270).*

## Sets and Indices

$$\begin{align*}
& V = \{1,\dots,n\} && \text{Set of items, indexed by } i,j. \\
& G = (V, A, l) && \text{Directed valued graph encoding precedence / time-lag relations.} \\
& A && \text{Set of arcs } (i,j) \text{ between items.} \\
& \mathcal{L} = \{1,\dots,L\} && \text{Set of bin indices, indexed by } b. \\
& \mathcal{T} = \{1,\dots,T\} && \text{Set of time periods, indexed by } p.
\end{align*}$$

## Parameters

$$\begin{align*}
& W && \text{Bin capacity (positive).} \\
& w_i && \text{Weight of item } i \in V, \text{ with } w_i \le W. \\
& l_{i,j} \in \mathbb{Z} && \text{Time lag associated with arc } (i,j) \in A \text{ (may be positive, zero, or negative).} \\
& L && \text{Maximum number of bins usable per time period (positive integer; } L=\infty \text{ if } L \ge n). \\
& T \in \mathbb{Z}_+ && \text{Upper bound on the number of time periods required in an optimal solution.}
\end{align*}$$

## Decision Variables

$$\begin{align*}
& x_{i,b,p} \in \{0,1\} && 1 \text{ if item } i \in V \text{ is assigned to bin } (b,p),\ b \in \mathcal{L},\ p \in \mathcal{T}. \\
& u_{b,p} \in \{0,1\} && 1 \text{ if bin } (b,p) \text{ has at least one item assigned, } b \in \mathcal{L},\ p \in \mathcal{T}.
\end{align*}$$

## Objective

$$\begin{align}
\min \quad & \sum_{b \in \mathcal{L}} \sum_{p \in \mathcal{T}} u_{b,p} \tag{2a}
\end{align}$$

## Constraints

$$\begin{align}
\text{s.t.}\quad
& \sum_{b \in \mathcal{L}} \sum_{p \in \mathcal{T}} x_{i,b,p} = 1
    && \forall i \in V, \tag{2b} \\
& \sum_{i \in V} w_i\, x_{i,b,p} \leq W\, u_{b,p}
    && \forall b \in \mathcal{L},\ p \in \mathcal{T}, \tag{2c} \\
& l_{i,j} + \sum_{p \in \mathcal{T}} p \cdot \sum_{b \in \mathcal{L}} x_{i,b,p}
    \leq \sum_{p \in \mathcal{T}} p \cdot \sum_{b \in \mathcal{L}} x_{j,b,p}
    && \forall (i,j) \in A, \tag{2d} \\
& x_{i,b,p} \in \{0,1\}
    && \forall i \in V,\ b \in \mathcal{L},\ p \in \mathcal{T}, \tag{2e} \\
& u_{b,p} \in \{0,1\}
    && \forall b \in \mathcal{L},\ p \in \mathcal{T}. \tag{2f}
\end{align}$$

## Preprocessing (not part of the formulation)

Let $d(i,j)$ be the length of the longest path from $i$ to $j$ in $G$ ($d(i,j)=-\infty$ if no path exists). Adding a source $s$ (time $0$) and sink $f$ (time $T+1$) with arcs $(s,i)$, $(i,f)$ of lag one gives $es(i)=d(s,i)$ and $ls(i)=T+1-d(i,f)$. Variables $x_{i,b,p}$ with $p<es(i)$ or $p>ls(i)$ may be fixed to zero.
