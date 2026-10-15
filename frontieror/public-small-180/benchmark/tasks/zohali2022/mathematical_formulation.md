# Original Formulation: Setup Assembly Line Balancing Problem, Type 2 (SUALBP-2 / SSBF-2)

*Source: Second station-based formulation (SSBF-2) for SUALBP-2, Zohali et al., 2022.* *Transcribed as presented (with the notation of Esmaeilbeigi et al. 2016, Table 1) in: “Domain-Independent Dynamic Programming and Constraint Programming Approaches for Assembly Line Balancing Problems with Setups,” J. Zhang and J. C. Beck, INFORMS Journal on Computing, Section 2.3, Eqs. (2a)–(2h).*

## Sets and Indices

$$\begin{align*}
& n && \text{number of tasks} \\
& V = \{1,2,\dots,n\} && \text{set of tasks; } i,j,v \text{ are task indices} \\
& \mathcal{E} && \text{set of precedence relations; } (i,j)\in\mathcal{E} \text{ if } i \text{ precedes } j \\
& P_i\ (P_i^{*}) && \text{set of direct (all) predecessors of task } i\in V \\
& F_i\ (F_i^{*}) && \text{set of direct (all) successors of task } i\in V \\
& E_i,\, L_i && \text{earliest / latest assignable station for task } i,\ E_i=\left\lceil (t_i+\textstyle\sum_{j\in P_i^{*}} t_j)/\bar{c}\right\rceil,\ L_i=\bar{m}+1-\left\lceil (t_i+\textstyle\sum_{j\in F_i^{*}} t_j)/\bar{c}\right\rceil \\
& KD,\, KP && \text{set of definite } (\{1,\dots,\underline{m}\}) \text{ and possible } (\{\underline{m}+1,\dots,\bar{m}\}) \text{ stations} \\
& K = KD \cup KP && \text{set of stations; } k \text{ is the station index} \\
& FS_i = \{E_i,\dots,L_i\} && \text{set of assignable stations for task } i\in V \\
& FT_k = \{i\in V \mid k\in FS_i\} && \text{set of tasks assignable to station } k\in K \\
& A_i = \{j\in V \mid FS_j\cap FS_i=\emptyset\} && \text{tasks that cannot share a station with } i \\
& F_i^{F}\ (P_i^{F}) && \text{tasks that can directly follow (precede) } i \text{ in forward direction},\ F_i^{F}=\{j\in V-(F_i^{*}-F_i)-P_i^{*}-A_i-\{i\}\},\ P_i^{F}=\{j\mid i\in F_j^{F}\} \\
& F_i^{B}\ (P_i^{B}) && \text{tasks that can directly follow (precede) } i \text{ in backward direction},\ F_i^{B}=\{j\in V-F_i^{*}-A_i\},\ P_i^{B}=\{j\mid i\in F_j^{B}\}
\end{align*}$$

## Parameters

$$\begin{align*}
& t_i && \text{execution (processing) time of task } i\in V \\
& \tau_{ij} && \text{forward setup time from task } i\in V \text{ to task } j\in F_i^{F} \text{ (triangle inequality holds)} \\
& \mu_{ij} && \text{backward setup time from task } i\in V \text{ to task } j\in F_i^{B} \text{ (no triangle inequality)} \\
& \bar{c}\ (\underline{c}) && \text{upper (lower) limit of the cycle time (from preprocessing)} \\
& \bar{m}\ (\underline{m}) && \text{upper (lower) limit of the station number; for SUALBP-2, } \bar{m}=\underline{m}=m \text{ (given)} \\
& D_{ij} && \text{lower bound on the cycle time } c \text{ if tasks } i \text{ and } j \text{ are at the same station (Esmaeilbeigi et al.\ 2016)}
\end{align*}$$

## Decision Variables

$$\begin{align*}
& x_{ik}\in\{0,1\} && =1 \text{ iff task } i\in V \text{ is assigned to station } k\in FS_i \\
& z_i\in\mathbb{Z}^{+} && \text{index of the station to which task } i\in V \text{ is assigned} \\
& g_{ijk}\in\{0,1\} && =1 \text{ iff task } i \text{ is performed immediately before task } j \text{ on station } k \text{ (forward sequencing)} \\
& h_{ijk}\in\{0,1\} && =1 \text{ iff task } i \text{ is the last and task } j \text{ is the first task on station } k \text{ (backward setup)} \\
& r_i\in\mathbb{Z}^{+} && \text{rank of task } i \text{ in the global sequence over all active stations} \\
& c && \text{continuous variable representing the cycle time}
\end{align*}$$

## Objective

$$\begin{align}
\min \quad & c \tag{2a}
\end{align}$$

## Constraints

$$\begin{align}
\text{s.t.}\quad
& \sum_{k\in FS_i} x_{ik} = 1, && \forall i\in V, \tag{1b} \\
& \sum_{k\in FS_i} k\cdot x_{ik} = z_i, && \forall i\in V, \tag{1c} \\
& \sum_{j\in FT_k\cap F_i^{F}} g_{ijk} + \sum_{j\in FT_k\cap F_i^{B}} h_{ijk} = x_{ik}, && \forall i\in V,\ \forall k\in FS_i, \tag{1d} \\
& \sum_{i\in FT_k\cap P_j^{F}} g_{ijk} + \sum_{i\in FT_k\cap P_j^{B}} h_{ijk} = x_{jk}, && \forall j\in V,\ \forall k\in FS_j, \tag{1e} \\
& r_i + 1 + \bigl(n-|F_i^{*}|-|P_j^{*}|\bigr)\Bigl(\sum_{k\in FS_i\cap FS_j} g_{ijk} - 1\Bigr) \le r_j, && \forall i\in V,\ \forall j\in F_i^{F}, \tag{1h} \\
& r_i + 1 \le r_j, && \forall (i,j)\in\mathcal{E}, \tag{1i} \\
& z_i \le z_j, && \forall (i,j)\in\mathcal{E}, \tag{1j} \\
& g_{ijk}\in\{0,1\}, && \forall k\in K,\ \forall i\in FT_k,\ \forall j\in (FT_k\cap F_i^{F}), \tag{1o} \\
& h_{ijk}\in\{0,1\}, && \forall k\in K,\ \forall i\in FT_k,\ \forall j\in (FT_k\cap F_i^{B}), \tag{1p} \\
& |P_i^{*}|+1 \le r_i \le n-|F_i^{*}|, && \forall i\in V, \tag{1q} \\
& x_{ik}\in\{0,1\}, && \forall i\in V,\ \forall k\in FS_i, \tag{1r} \\
& r_i,\, z_i \in \mathbb{Z}^{+}, && \forall i\in V, \tag{1s} \\[4pt]
% --- SSBF-2-specific rows ---
& \sum_{i\in FT_k} \sum_{j\in (FT_k\cap F_i^{B})} h_{ijk} = 1, && \forall k\in K, \tag{2c} \\
& \sum_{i\in FT_k} t_i\,x_{ik} + \sum_{i\in FT_k} \sum_{j\in (FT_k\cap F_i^{F})} \tau_{ij}\,g_{ijk} + \sum_{i\in FT_k\cap P_i^{B}} \mu_{ij}\,h_{ijk} \le c, && \forall k\in K, \tag{2d} \\
& \sum_{i\in FT_k\setminus\{j\}} x_{ik} \le (n-\underline{m}+1)(1-h_{jjk}), && \forall k\in K,\ \forall j\in FT_k, \tag{2e} \\
& \sum_{i\in FT_k} x_{ik} \ge 1, && \forall k\in K, \tag{2f} \\
& c + \bar{c}\Bigl(\sum_{k\in FS_j} k\cdot x_{jk} - \sum_{k\in FS_i} k\cdot x_{ik}\Bigr) \ge D_{ij}, && \forall i\in V,\ j\in (F_i^{F}\setminus A_i), \tag{2g} \\
& \underline{c} \le c \le \bar{c}. && \tag{2h}
\end{align}$$
