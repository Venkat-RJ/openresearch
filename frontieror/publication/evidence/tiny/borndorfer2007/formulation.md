# Original Formulation: Line Planning Problem (LPP)

*Source: A Column-Generation Approach to Line Planning in Public Transport, Ralf Borndörfer, Martin Grötschel, Marc E. Pfetsch, 2007 (Transportation Science 41(1), pp. 123–132).*

## Sets and Indices

- $M$ — number of transportation *modes* (bus, tram, subway, etc.), indexed by $i = 1,\dots,M$.

- $G = (V,E) = (V,\, E_1 \,\dot\cup\, \cdots \,\dot\cup\, E_M)$ — undirected multigraph representing the multimodal transportation network ($V$ nodes, $E$ edges partitioned by mode).

- $G_i = (V, E_i)$ — subnetwork of $G$ corresponding to mode $i$.

- $\mathcal{T}_1,\dots,\mathcal{T}_M \subseteq V$ — terminal sets; nodes where lines of each mode can start and end.

- $\mathcal{L}$ — set of all feasible lines. A *line* of mode $i$ is a (simple) path in $G_i$ connecting two different terminals of $\mathcal{T}_i$.

- $\mathcal{L}_e := \bigcup\{\, l \in \mathcal{L} : e \in l \,\}$ — set of lines that use edge $e \in E$.

- $D := \{\, (s,t) \in V \times V : d_{st} > 0 \,\}$ — set of all OD-pairs.

- $(V, A)$ — directed passenger route graph derived from $G = (V,E)$ by replacing each edge $e \in E$ with two antiparallel arcs $a(e)$ and $\bar a(e)$; $e(a) \in E$ denotes the undirected edge corresponding to $a \in A$.

- $\mathcal{P}_{st}$ — set of all $(s,t)$-passenger paths (directed $s$–$t$ paths in $(V,A)$), for $(s,t) \in D$.

- $\mathcal{P} := \bigcup\{\, p \in \mathcal{P}_{st} : (s,t) \in D \,\}$ — set of all passenger paths.

- $\mathcal{P}_a := \bigcup\{\, p \in \mathcal{P} : a \in p \,\}$ — set of all passenger paths using arc $a \in A$.

## Parameters

- $d_{st} \in \mathbb{Q}_+$ — travel demand from $s$ to $t$ (entry of the not-necessarily-symmetric OD-matrix), for $(s,t) \in V \times V$.

- $\mathbf{c}^i \in \mathbb{Q}_+^{E_i}$ — line operating costs on the edges of mode $i$; $c_l := \sum_{e \in l} c^i_e$ is the operating cost of line $l$ of mode $i$.

- $C_1,\dots,C_M \in \mathbb{Q}_+$ — fixed cost for setting up a line of each mode; $C_l := C_i$ for a line $l$ of mode $i$.

- $\kappa_1,\dots,\kappa_M \in \mathbb{Q}_+$ — vehicle capacity for each mode; $\kappa_l := \kappa_i$ for a line $l$ of mode $i$.

- $\Lambda \in \mathbb{Q}_+^{E}$ — edge (frequency) capacities; $\Lambda_e$ bounds the total frequency of lines using $e \in E$.

- $\boldsymbol{\tau} \in \mathbb{Q}_+^{A}$ — traveling time on arc $a \in A$; $\tau_p := \sum_{a \in p} \tau_a$ is the traveling time on path $p$.

- $F$ — global upper bound on the frequency of any single line, assumed $F \ge \Lambda_e$ for all $e \in E$.

## Decision Variables

- $y_p \in \mathbb{R}_+$ — flow of passengers traveling from $s$ to $t$ on path $p \in \mathcal{P}_{st}$.

- $f_l \in \mathbb{R}_+$ — frequency of line $l \in \mathcal{L}$.

- $x_l \in \{0,1\}$ — decision variable for using line $l \in \mathcal{L}$.

## Objective

$$\text{(LPP)}\qquad
\min \ \boldsymbol{\tau}^{\mathsf T}\mathbf{y} + \mathbf{C}^{\mathsf T}\mathbf{x} + \mathbf{c}^{\mathsf T}\mathbf{f}$$ where $\boldsymbol{\tau}^{\mathsf T}\mathbf{y}$ is total passenger traveling time, $\mathbf{C}^{\mathsf T}\mathbf{x}$ is the fixed line set-up cost, and $\mathbf{c}^{\mathsf T}\mathbf{f}$ is the variable operating cost of lines at frequencies $\mathbf{f}$.

## Constraints

Using the notation $\mathbf{v}(I) := \sum_{i \in I} v_i$ for a vector $\mathbf{v}$ indexed by a set $I$: $$\begin{align}
  \mathbf{y}(\mathcal{P}_{st}) &= d_{st} & &\forall\,(s,t) \in D \tag{1}\\[2pt]
  \mathbf{y}(\mathcal{P}_a) - \sum_{l:\, e(a) \in l} \kappa_l f_l &\le 0 & &\forall\, a \in A \tag{2}\\[2pt]
  \mathbf{f}(\mathcal{L}_e) &\le \Lambda_e & &\forall\, e \in E \tag{3}\\[2pt]
  \mathbf{f} &\le F\mathbf{x} & & \tag{4}\\[2pt]
  x_l &\in \{0,1\} & &\forall\, l \in \mathcal{L} \tag{5}\\[2pt]
  f_l &\ge 0 & &\forall\, l \in \mathcal{L} \tag{6}\\[2pt]
  y_p &\ge 0 & &\forall\, p \in \mathcal{P}. \tag{7}
\end{align}$$ Constraints (1) and (7) model a multicommodity flow problem routing the demand; (2) are the capacity constraints linking passenger paths with line capacity on each arc; (3) are the frequency (edge-capacity) constraints; (4) are the linking constraints forcing $f_l = 0$ whenever line $l$ is not used ($f_l \le F x_l$).
