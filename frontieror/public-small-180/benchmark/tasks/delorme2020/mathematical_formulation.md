# Original Formulation: Cutting Stock Problem – Reflect ($F_{RE}$)

*Source: Enhanced Pseudo-polynomial Formulations for Bin Packing and Cutting Stock Problems, Maxence Delorme, Manuel Iori, 2019 (INFORMS Journal on Computing).*

The paper studies the bin packing problem (BPP) and the cutting stock problem (CSP) and proposes a new pseudo-polynomial formulation, *reflect* ($F_{RE}$, Section 4), which models a CSP instance using only half of the bin capacity. $F_{RE}$ is the formulation whose notation is carried into the subsequent algorithmic ($reflect+$) and computational sections, and is reported below as the canonical model. The classical arc-flow formulation $F_{AF}$ (Section 2.3), on which reflect builds, is given under *Variants*.

## Sets and Parameters

$$\begin{align*}
c        & : \text{bin (stock) capacity, integer.} \\
m        & : \text{number of item types.} \\
w_j      & : \text{width of item type } j,\ j=1,\dots,m,\ \text{integer, } 0 < w_j < c. \\
d_j      & : \text{demand of item type } j,\ j=1,\dots,m,\ \text{integer.} \\
\mathcal{S} & : \text{set of feasible item-width combinations not exceeding } c, \\
            & \quad \mathcal{S}=\Big\{\hat{w}=\textstyle\sum_{j=1}^{m} w_j x_j : \hat{w}\le c,\ x_j\in\mathbb{N},\ j=1,\dots,m\Big\}. \tag{3}\\
R          & : \text{reflection vertex corresponding to } c/2 \text{ (added explicitly; may be fractional if } c \text{ is odd).}
\end{align*}$$

The formulation is built on a multigraph $\mathcal{G}=(\mathcal{V},\mathcal{A})$: $$\begin{align*}
\mathcal{V} & : \text{set of vertices; all partial bin fillings of size smaller than } c/2,\\
            & \quad \text{plus the additional vertex } R \text{ corresponding to } c/2. \\
\mathcal{A} & : \text{set of arcs, partitioned as } \mathcal{A}=\mathcal{A}_s \cup \mathcal{A}_r. \\
\mathcal{A}_s & : \text{standard arcs (item and loss arcs proceeding left to right as in } F_{AF}\text{);}\\
              & \quad \text{each is the triplet } (d,e,s). \\
\mathcal{A}_r & : \text{reflected arcs, i.e. item arcs } (d,e) \text{ of } F_{AF} \text{ with } d<c/2,\, e>c/2 \\
              & \quad \text{reflected into } (d,c-e); \text{ plus the collision arc } (c/2,c/2,r)=(R,R,r);\\
              & \quad \text{each is the triplet } (d,e,r). \\
(d,e,\kappa) & : \text{generic arc with } \kappa\in\{s,r\}. \\
\mathcal{A}_j & : \text{item arcs associated with item type } j,\\
              & \quad \mathcal{A}_j=\{(d,d+w_j,s)\in\mathcal{A}_s\}\cup\{(d,c-d-w_j,r)\in\mathcal{A}_r\}. \\
\delta_s^-(e) & \subseteq \mathcal{A}_s : \text{standard arcs entering vertex } e. \\
\delta_r^-(e) & \subseteq \mathcal{A}_r : \text{reflected arcs entering vertex } e. \\
\delta^+(e) & : \text{arcs emanating from vertex } e.
\end{align*}$$

## Decision Variables

$$\begin{align*}
z              & : \text{number of bins used (integer).} \\
\xi_{de\kappa} & : \text{number of times arc } (d,e,\kappa)\in\mathcal{A} \text{ is chosen, integer.}
\end{align*}$$

## Objective

$$\begin{align}
(F_{RE}) \quad \min \quad & z = \sum_{(d,e,r) \in \mathcal{A}_r} \xi_{der}
    \tag{16}
\end{align}$$

## Constraints

$$\begin{align}
& \sum_{(d,e,s) \in \delta_s^-(e)} \xi_{des}
   = \sum_{(d,e,r) \in \delta_r^-(e)} \xi_{der}
   + \sum_{(e,f,\kappa) \in \delta^+(e)} \xi_{ef\kappa},
   && e \in \mathcal{V} \setminus \{0\}, \tag{17} \\
& \sum_{(0,e,\kappa) \in \delta^+(0)} \xi_{0e\kappa}
   = 2 \sum_{(d,e,r) \in \mathcal{A}_r} \xi_{der}, \tag{18} \\
& \sum_{(d,e,\kappa) \in \mathcal{A}_j} \xi_{de\kappa} \geq d_j,
   && j = 1, \ldots, m, \tag{19} \\
& \xi_{de\kappa} \in \mathbb{N},
   && (d,e,\kappa) \in \mathcal{A}. \tag{20}
\end{align}$$
