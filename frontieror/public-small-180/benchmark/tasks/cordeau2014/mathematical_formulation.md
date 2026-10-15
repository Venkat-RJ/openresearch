# Original Formulation: Time-Dependent Travelling Salesman Problem (TDTSP)

*Source: Analysis and Branch-and-Cut for the Time-Dependent Travelling Salesman Problem, Cordeau, Ghiani, and Guerriero, 2014.*

## Sets and Parameters

- $G = (V \cup \{0\}, A)$: complete directed graph with $V = \{1,\dots,n\}$ and depot $0$.

- $[0, T]$: time horizon partitioned into $H$ subintervals $[T_h, T_{h+1}]$, $h = 0,\dots,H-1$, with $T_0 = 0$ and $T_H = T$.

- $L_{ij}$: length of arc $(i,j) \in A$.

- $u_{ij} = \max_{h} v_{ijh}$: maximum (free-flow) speed on arc $(i,j)$.

- $v_{ijh} = \delta_{ijh}\, b_h\, u_{ij}$: actual speed on $(i,j)$ during period $h$, where $b_h \in [0,1]$ is the common congestion factor and $\delta_{ijh} \in [0,1]$ is an arc-specific degradation.

- $\mathcal{P}_i$: set of simple paths $\{i_0, i_1, \dots, i_m\}$ on $G$ with $i_0 = 0$ and $i_m = i$.

- $LB_{\mathfrak{p}}$: lower bound on total tour duration when path $\mathfrak{p} \in \mathcal{P}_i$ is a prefix of the tour (computed via Algorithm 2 of the paper).

- $\underline{z}(\underline{c}^{*})$: lower bound given by the optimal ATSP tour with arc costs $L_{ij}/u_{ij}$ (speed law $v_{ijh} = b_h u_{ij}$).

## Decision Variables

- $x_{ij} \in \{0,1\}$, $(i,j) \in A$: $=1$ iff arc $(i,j)$ is in the tour.

- $z \in \mathbb{R}$: total tour duration.

- $z_h \geq 0$, $h = 0,\dots,H-1$: auxiliary continuous variable representing the portion of tour duration attributable to period $h$ (under the common-congestion speed law).

- $y_h \in \{0,1\}$, $h = 0,\dots,H-1$: $=1$ iff the tour completion time lies in $[T_{h-1}, T_h]$.

## Objective

$$\begin{equation}
\min \;\; z \tag{12}
\end{equation}$$

## Constraints (Core Model, Section 4)

$$\begin{align}
z &\;\geq\; LB_{\mathfrak{p}} \Bigl( 1 - \sum_{(a,b) \in \mathfrak{p}} (1 - x_{ab}) \Bigr),
   & \forall i \in V \cup \{0\},\; \mathfrak{p} \in \mathcal{P}_i \tag{13} \\[2pt]
\sum_{i \in V \cup \{0\} \setminus \{j\}} x_{ij} &= 1, & \forall j \in V \cup \{0\} \tag{14} \\[2pt]
\sum_{j \in V \cup \{0\} \setminus \{i\}} x_{ij} &= 1, & \forall i \in V \cup \{0\} \tag{15} \\[2pt]
\sum_{i \in S} \sum_{j \notin S} x_{ij} &\;\geq\; 1,
   & \forall S \subset V \cup \{0\},\; |S| \geq 2 \tag{16} \\[2pt]
x_{ij} &\;\in\; \{0,1\}, & \forall (i,j) \in A \tag{17}
\end{align}$$

## Valid Inequalities (Section 4.1)

$$\begin{align}
z &\;\geq\; \sum_{h=0}^{H-1} \frac{z_h}{b_h} \tag{20} \\[2pt]
z_0 &\;\geq\; \sum_{(i,j) \in A} \frac{L_{ij}}{u_{ij}}\, x_{ij}
        \;-\; \sum_{\ell=1}^{H-1} z_\ell \tag{21} \\[2pt]
z_h &\;\geq\; \sum_{(i,j) \in A} \frac{L_{ij}}{u_{ij}}\, x_{ij}
        \;-\; \sum_{\ell=0}^{h-1} b_\ell\,(T_\ell - T_{\ell-1})
        \;-\; \sum_{\ell=h+1}^{H-1} z_\ell,
   & h = 1, \dots, H-2 \tag{22} \\[2pt]
z_{H-1} &\;\geq\; \sum_{(i,j) \in A} \frac{L_{ij}}{u_{ij}}\, x_{ij}
             \;-\; \sum_{\ell=0}^{H-2} b_\ell\,(T_\ell - T_{\ell-1}) \tag{23} \\[2pt]
z_h &\;\leq\; b_h\,(T_h - T_{h-1}) \sum_{\ell=h}^{H-1} y_\ell,
   & h = 0, \dots, H-1 \tag{24} \\[2pt]
z_h &\;\geq\; b_h\,(T_h - T_{h-1}) \sum_{\ell=h+1}^{H-1} y_\ell,
   & h = 0, \dots, H-2 \tag{25} \\[2pt]
\sum_{h=0}^{H-1} y_h &= 1 \tag{26} \\[2pt]
z_h \;\geq\; 0, \;\; y_h &\;\in\; \{0,1\},
   & h = 0, \dots, H-1 \tag{27}
\end{align}$$

## Initial Bound (eq. 31)

$$\begin{equation}
z \;\geq\; \underline{z}(\underline{c}^{*}) \tag{31}
\end{equation}$$

Constraints (13) and (16) are exponential in size and are separated dynamically (shortest-path separation for (13); min-cut separation for (16)). Valid inequalities (20)–(27), together with (31), are added up-front as the total count $3H + 1$ is small.
