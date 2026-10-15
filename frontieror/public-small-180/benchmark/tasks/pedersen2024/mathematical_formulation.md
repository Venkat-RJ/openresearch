# Original Formulation: Quota Steiner Tree Problem (QSTP)

*Source: An efficient solver for large-scale onshore wind farm siting including cable routing, Jaap Pedersen, Jann Michael Weinand, Chloi Syranidou, Daniel Rehfeldt, 2024.*

## Sets and Indices

$$\begin{align*}
G = (V,E)            &: \text{undirected graph (turbine sites, substations, Steiner nodes).}\\
T_f \subset V        &: \text{set of fixed terminals (grid substations).}\\
T_p \subset V        &: \text{set of potential terminals (turbine positions), } T_f \cap T_p = \emptyset.\\
v \in V \setminus (T_f \cup T_p) &: \text{Steiner node.}\\
D = (V,A)            &: \text{directed graph with } A := \{(i,j),(j,i)\mid \forall (i,j)\in E\}.\\
r \in T_f            &: \text{arbitrary root node chosen among the fixed terminals.}\\
\delta^+(W)          &= \{(i,j)\in A : i\in W,\, j\in V\setminus W\} \quad \text{(outgoing arcs of } W).\\
\delta^-(W)          &= \{(i,j)\in A : i\in V\setminus W,\, j\in W\} \quad \text{(incoming arcs of } W).
\end{align*}$$ For a single vertex $v_i$ we write $\delta^+(v_i):=\delta^+(\{v_i\})$ and $\delta^-(v_i):=\delta^-(\{v_i\})$. For any arc set $K$, $x(K):=\sum_{(i,j)\in K} x_{ij}$.

## Parameters

$$\begin{align*}
c : E \to \mathbb{R}_{\geq 0} &: \text{cable (edge) cost; } c_e \text{ cost of edge } e=(i,j).\\
w : T_p \to \mathbb{R}_{>0}   &: \text{vertex (turbine installation) cost.}\\
q : T_p \to \mathbb{R}_{>0}   &: \text{quota profit (annual energy yield) of a potential terminal.}\\
Q \in \mathbb{R}_{>0}         &: \text{minimum quota (minimum total energy yield required).}
\end{align*}$$ Vertex costs are shifted onto incoming arc costs, giving directed arc costs $c:A\to\mathbb{R}_{\geq 0}$: $$\begin{equation}
c(i,j)=
\begin{cases}
c_e + w_j & \text{if } j \in T_p,\\
c_e       & \text{otherwise,}
\end{cases}
\qquad \forall a=(i,j)\in A. \tag{3}
\end{equation}$$

## Decision Variables

$$\begin{align*}
x_{ij} &\in \{0,1\} : \text{1 if arc } (i,j)\in A \text{ is in the Steiner arborescence, 0 otherwise.}\\
y_{k}  &\in \{0,1\} : \text{1 if potential terminal } v_k \in T_p \text{ is chosen, 0 otherwise.}
\end{align*}$$

## Objective

The QSTP seeks a tree $S=(V',E')\subseteq G$ containing all fixed terminals $T_f$ that minimizes the total cable and turbine cost $$\begin{equation}
C(S) = \sum_{(i,j)\in E'} c_{ij} + \sum_{i \in T_p \cap V'} w_i \tag{1}
\end{equation}$$ subject to a fulfilled quota $$\begin{equation}
Q(S)=\sum_{i\in T_p\cap V'} q_i \geq Q. \tag{2}
\end{equation}$$ Modeled as a rooted Steiner arborescence problem with the shifted arc costs $c$ of $(3)$, the objective is $$\begin{equation}
\min \quad c^T x. \tag{4}
\end{equation}$$

## Constraints

$$\begin{align}
x(\delta^-(W)) &\geq 1
   && \forall W \subset V,\; r \notin W,\; |W \cap T_f| \geq 1, \tag{5}\\[2pt]
x(\delta^-(W)) &\geq y_i
   && \forall W \subset V,\; r \notin W,\; |W \cap T_p| \geq 1,\; v_i \in T_p, \tag{6}\\[2pt]
\sum_{v_i \in T_p} q_i\, y_i &\geq Q, && \tag{7}\\[2pt]
x_{ij},\, y_k &\in \{0,1\}
   && \forall (i,j) \in A,\; \forall v_k \in T_p. \tag{8}
\end{align}$$
