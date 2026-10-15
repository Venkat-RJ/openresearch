# Original Formulation: Integrated Timetabling and Vehicle Scheduling (ITTVS)

*Source: A matheuristic for integrated timetabling and vehicle scheduling, S. Carosi, A. Frangioni, L. Galli, L. Girardi, G. Vallese, Transportation Research Part B 127 (2019) 99–124.*

The model below is the integrated MILP presented in Section 4.3 (“TT–VS integrated model”), equations (1)–(6), whose notation is carried into the solution approach (Section 6) and the computational results (Section 7). The TT sub-problem (graph $G_d^{TT}$, Section 4.1) and the VS sub-problem (graph $G^{VS}$, Section 4.2) are coupled by the linking constraints (6).

## Sets and Indices

- $\mathcal{D}$ : set of directions $d$ (e.g. $\mathcal{D}=\{\overrightarrow{AB},\overrightarrow{BA}\}$ for a simple single line).

- $\mathcal{T}=[\mathcal{T}_d]_{d\in\mathcal{D}}$ : set of all potential trips; $\mathcal{T}_d\subset\mathcal{T}$ is the subset of potential trips of direction $d$.

- $\mathcal{T}_d^{ini},\,\mathcal{T}_d^{fin}\subseteq\mathcal{T}_d$ : admissible initial / final trips for direction $d$.

- $G_d^{TT}=(N_d^{TT},A_d^{TT})$ : (acyclic) TT compatibility graph of direction $d$, with $N_d^{TT}=\mathcal{T}_d\cup\{O_d^-,O_d^+\}$, where $O_d^-$ is a dummy source and $O_d^+$ a dummy sink. An arc $(i,j)\in A_d^{TT}$ exists iff the headway $a(j)-a(i)$ is feasible (Section 4.1).

- $G^{VS}=(N^{VS},A^{VS})$ : (single) VS compatibility graph, with two nodes $i^-,i^+$ per trip $i\in\mathcal{T}$ plus depot nodes $O^-,O^+$ (“pure” version, Section 4.2.1; see *Variants* for the time-space version).

- $B(i)$ : backward star of trip node $i$ in $N_{d(i)}^{TT}$, i.e. the set of TT arcs entering node $i$.

- $N=\{A,B\}$ : terminals of the line; $N^+=N\cup\{O\}$ : terminals plus the single depot $O$.

- $d(i)$ : direction of trip $i$; $h(i)$ : time window in which trip $i$ falls.

## Parameters

- $b_n^d$ : node deficit in $G_d^{TT}$; $b_{O_d^-}^d=-1$, $b_{O_d^+}^d=+1$, and $b_n^d=0$ otherwise.

- $c^d$ : (row) cost vector of TT arcs in $A_d^{TT}$; the cost of arc $(i,j)$ is the headway-deviation penalty of $a(j)-a(i)$ from the ideal headway $\hat{l}_d^{\,h(i)}$ (zero at the ideal headway; a quadratic penalty in the experiments).

- $c$ : (row) cost vector of VS arcs in $A^{VS}$: trip arcs $(i^-,i^+)$ cost $0$; in-line arcs $(i^+,j^-)$ cost $st(j)-et(i)-\underline{\delta}_{en(i)}^{\,h(i)}$ (extra waiting); out-line arcs $(i^+,j^-)$ cost $t_{en(i),-}^{\,h(i)}+t_{sn(j),+}^{\,h(j)}$; start arcs $(O^-,i^-)$ cost $t_{sn(i),+}^{\,h(i)}$; end arcs $(i^+,O^+)$ cost $t_{en(i),-}^{\,h(i)}$; the return arc $(O^+,O^-)$ has a large cost $M$ (per-vehicle deployment cost).

- $u_{n,m}$ : capacity of VS arc $(n,m)\in A^{VS}$; $u_{n,m}=1$ for trip, in-line/out-line, start and end arcs; $u_{O^+,O^-}=K$ (maximum fleet cardinality, or $+\infty$ if unconstrained).

- $\alpha$ : weighting parameter balancing VS cost against TT quality (primary trade-off parameter).

- $\hat{l}_d^{\,h},\ \underline{l}_d^{\,h},\ \bar{l}_d^{\,h}$ : ideal, minimum and maximum headway of direction $d$ in window $h$.

- $a(i)$ : arrival time of trip $i$ at its main stop; $st(i),et(i)$ : departure / arrival time of trip $i$ at $sn(i),en(i)$.

- $sn(i),en(i)$ : start and end terminal of trip $i$.

- $\underline{\delta}_n^{\,h},\ \bar{\delta}_n^{\,h}$ : minimum / maximum stopping time at terminal $n\in N^+$ in window $h$ ($\bar{\delta}_O^{\,h}=+\infty$).

- $t_{n,+}^{\,h},\ t_{n,-}^{\,h}$ : pull-out (depot$\to n$) and pull-in ($n\to$depot) travel times of terminal $n$ in window $h$.

## Decision Variables

- $y_{n,m}^d\in\{0,1\}$ : TT arc-selection variable, $(n,m)\in A_d^{TT}$, $d\in\mathcal{D}$; equals $1$ iff arc $(n,m)$ belongs to the chosen timetable path $\pi_d$ (i.e. the $O_d^-$–$O_d^+$ path).

- $x_{n,m}\ge 0$ : VS arc-flow variable, $(n,m)\in A^{VS}$ (the flow defining the vehicle schedule $\Omega$). It need not be declared integer: by total unimodularity of (4)–(5) it is automatically integral once $y$ is integral.

## Objective

$$\begin{equation}
\min \quad \alpha\, c\,x + \sum_{d\in\mathcal{D}} c^d y^d \tag{1}
\end{equation}$$

## Constraints

$$\begin{align}
& \sum_{(m,n)\in A_d^{TT}} y_{m,n}^d \;-\; \sum_{(n,m)\in A_d^{TT}} y_{n,m}^d \;=\; b_n^d,
   && n\in N_d^{TT},\ d\in\mathcal{D} \tag{2}\\[2pt]
& y_{n,m}^d \in \{0,1\}, && (n,m)\in A_d^{TT},\ d\in\mathcal{D} \tag{3}\\[2pt]
& \sum_{(m,n)\in A^{VS}} x_{m,n} \;-\; \sum_{(n,m)\in A^{VS}} x_{n,m} \;=\; 0,
   && n\in N^{VS} \tag{4}\\[2pt]
& 0 \le x_{n,m} \le u_{n,m}, && (n,m)\in A^{VS} \tag{5}\\[2pt]
& \sum_{(n,m)\in B(i)} y_{n,m}^{\,d(i)} \;=\; x_{i^-,\,i^+}, && i\in\mathcal{T} \tag{6}
\end{align}$$

Constraints (2)–(3) are the flow-conservation/integrality constraints of the $|\mathcal{D}|$ TT sub-problems (one $O_d^-$–$O_d^+$ path per direction, i.e. timetable $\pi_d$). Constraints (4)–(5) are the flow-conservation and capacity constraints of the VS sub-problem (a circulation on $G^{VS}$). Constraints (6) are the linking constraints: trip $i$ is selected in the timetable (flow on a TT arc entering node $i$) iff its trip arc $(i^-,i^+)$ carries flow in the vehicle schedule.
