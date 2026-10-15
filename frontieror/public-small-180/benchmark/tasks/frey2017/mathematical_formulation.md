# Original Formulation: Outbound Baggage Handling Problem (OBHP) — Time-Indexed Formulation (TIF)

*Source: Column Generation for Outbound Baggage Handling at Airports, Markus Frey, Rainer Kolisch, Christian Artigues, Transportation Science (2017).*

The paper states (Section 3) that it is sufficient to solve a Time-Indexed Formulation (TIF), presented in Section 3.1 as equations (1)–(9). This is the formulation that defines the problem the paper studies; its notation ($x_{i,c,w,\tau}$, $y_{c,k,m}$) is carried into the structural analysis, preprocessing, and computational experiments. The Dantzig–Wolfe master problem (10)–(14) and the pricing/scheduling subproblems are an *algorithmic* reformulation introduced in Section 4 and are therefore excluded here.

## Sets and Indices

$$\begin{align*}
\mathcal{F} &= \{1,\dots,F\} && \text{set of outgoing flights; index } i\\
\mathcal{T} &= \{t_0,\dots,t_T\},\ t_0=0<t_1<\dots<t_T && \text{discrete planning horizon (evenly spaced); index } t\\
& && \quad t_k \text{ denotes period } [t_k,t_{k+1}[\\
\mathcal{C} &= \{1,\dots,C\} && \text{set of carousels; index } c\\
\mathcal{R} &= \{1,\dots,R\} && \text{set of carousel types; index } r\\
\mathcal{C}_r &= \{1,\dots,C_r\},\ \mathcal{C}_{r_1}\cap\mathcal{C}_{r_2}=\varnothing\ (r_1\neq r_2) && \text{carousels of type } r;\ r_c \text{ is the type of carousel } c\\
\mathcal{U} &= \{u_1,\dots,u_K\},\ 0=u_0<u_1<\dots<u_K && \text{ordered threshold values for utilization deviation}\\
\mathcal{T}_1,\dots,\mathcal{T}_M,\ \bar{\mathcal{T}}=\textstyle\bigcup_{m=1}^{M}\mathcal{T}_m && M\ge 1\ \text{disjoint peak intervals; index } m\\
\mathcal{S}_i && \text{flight } i\text{'s feasible start time tuples } \tau=\langle s^h_\tau,s^d_\tau\rangle\\
\mathcal{S}_i(w) &\subseteq \mathcal{S}_i && \text{feasible tuples of } i \text{ when } w \text{ working stations are assigned}
\end{align*}$$

A start time tuple $\tau=\langle s^h_\tau,s^d_\tau\rangle$ records the start time $s^h_\tau$ of baggage handling at the carousel and the start time $s^d_\tau$ of storage depletion. $\tau$ is feasible for flight $i$ iff $s^h_\tau\in[S^{es}_i,S^{ls}_i]$ and $s^h_\tau\le s^d_\tau<S^e_i-\Delta(s^h_\tau)$, where $\Delta(s^h_\tau)$ is the storage-depletion offset.

## Parameters

$$\begin{align*}
K^{cb}_r,\ K^{pp}_r,\ K^{ws}_r,\ K^{ppws}_r &&& \text{type-}r\text{ conveyor-belt cap., parking positions, working stations,}\\
& &&\quad \text{parking positions per working station}\\
A_i=(A_{i,t})_{t=0,\dots,T-1} &&& \text{baggage arrival vector of flight } i\\
P_i &&& \text{number of required containers for flight } i\\
[W^{\min}_{i,r},W^{\max}_{i,r}] &&& \text{bounds on working stations for } i \text{ at carousel type } r\\
&&&\quad W^{\min}_{i,r}=\max\{\lfloor P_i/K^{ppws}_r\rfloor,1\}\\
&&&\quad W^{\max}_{i,r}=\lceil P_i/K^{ppws}_r\rceil+\mathbf{1}_{\{n\in\mathbb{N}:n>1\}}(P_i \bmod K^{ppws}_r)\\
[W^{\min}_i,W^{\max}_i] &&& \text{overall working-station bounds for } i \text{ across all types}\\
[S^{es}_i,S^{ls}_i] &&& \text{time window for the start of flight } i\text{'s baggage handling}\\
S^e_i &&& \text{end time of flight } i\text{'s baggage handling}\\
K^s &&& \text{capacity of the central storage system}\\
u^{ta}<1 &&& \text{target utilization}\\
p_k,\ 0<p_1<\dots<p_K &&& \text{penalty for a deviation in } ]u_{k-1},u_k]\ (\text{e.g., } p_k=k^k)\\
\Gamma^{i,w}_{t,\tau} &&& \text{flight } i\text{'s carousel workload in period } t,\ w \text{ stations, tuple } \tau\\
\Phi^{i}_{t,\tau} &&& \text{flight } i\text{'s baggage stored centrally in period } t \text{ for tuple } \tau
\end{align*}$$

*Remark.* $\Gamma^{i,w}_{t,\tau}$ and $\Phi^{i}_{t,\tau}$ are derived from $A_i$, $w$, and $\tau$; the paper gives their derivation in Online Appendix B (not in the main text).

## Decision Variables

$$\begin{align*}
x_{i,c,w,\tau}\in\{0,1\} &&& =1 \text{ if flight } i \text{ is processed at carousel } c \text{ with } w \text{ working}\\
&&&\quad \text{stations and start time tuple } \tau\\
&&&\quad \forall\, i\in\mathcal{F},\ c\in\mathcal{C},\ \tau\in\mathcal{S}_i,\ W^{\min}_{i,r_c}\le w\le W^{\max}_{i,r_c}\\
y_{c,k,m}\in\{0,1\} &&& =1 \text{ if utilization of carousel } c \text{ exceeds } u^{ta} \text{ by } u_k\\
&&&\quad \text{in peak interval } m,\quad \forall\, c\in\mathcal{C},\ 1\le k\le K,\ 1\le m\le M
\end{align*}$$

## Objective

$$\begin{equation}
\text{minimize}\quad \sum_{c\in\mathcal{C}}\ \sum_{1\le k\le K}\ \sum_{1\le m\le M} p_k\, y_{c,k,m} \tag{1}
\end{equation}$$

## Constraints

$$\begin{align}
& \sum_{c\in\mathcal{C}}\ \sum_{W^{\min}_{i,r}\le w\le W^{\max}_{i,r}}\ \sum_{\tau\in\mathcal{S}_i(w)} x_{i,c,w,\tau} = 1
&& \forall\, i\in\mathcal{F} \tag{2}\\[4pt]
& \sum_{\substack{i\in\mathcal{F}:\\ S^{es}_i\le t<S^e_i}}\ \sum_{W^{\min}_{i,r}\le w\le W^{\max}_{i,r}}\ \sum_{\substack{\tau\in\mathcal{S}_i(w):\\ s^h_\tau\le t}} w\, x_{i,c,w,\tau} \le K^{ws}_r
&& \forall\, r\in\mathcal{R},\, c\in\mathcal{C}_r,\, t\in\mathcal{T} \tag{3}\\[4pt]
& \sum_{\substack{i\in\mathcal{F}:\\ S^{es}_i\le t<S^e_i}}\ \sum_{W^{\min}_{i,r}\le w\le W^{\max}_{i,r}}\ \sum_{\substack{\tau\in\mathcal{S}_i(w):\\ s^h_\tau\le t}} P_i\, x_{i,c,w,\tau} \le K^{pp}_r
&& \forall\, r\in\mathcal{R},\, c\in\mathcal{C}_r,\, t\in\mathcal{T} \tag{4}\\[4pt]
& \sum_{\substack{i\in\mathcal{F}:\\ S^{es}_i\le t<S^e_i}}\ \sum_{c\in\mathcal{C}}\ \sum_{W^{\min}_{i,r}\le w\le W^{\max}_{i,r}}\ \sum_{\tau\in\mathcal{S}_i(w)} \Phi^{i}_{t,\tau}\, x_{i,c,w,\tau} \le K^{s}
&& \forall\, t\in\mathcal{T} \tag{5}\\[4pt]
& \frac{1}{K^{cb}_r}\sum_{\substack{i\in\mathcal{F}:\\ S^{es}_i\le t<S^e_i}}\ \sum_{W^{\min}_{i,r}\le w\le W^{\max}_{i,r}}\ \sum_{\tau\in\mathcal{S}_i(w)} \Gamma^{i,w}_{t,\tau}\, x_{i,c,w,\tau}
- \sum_{1\le k\le K} u_k\, y_{c,k,m} \le u^{ta}
&& \forall\, r\in\mathcal{R},\, c\in\mathcal{C}_r,\, 1\le m\le M,\, t\in\mathcal{T}_m \tag{6}\\[4pt]
& \sum_{1\le k\le K} y_{c,k,m} \le 1
&& \forall\, c\in\mathcal{C},\, 1\le m\le M \tag{7}\\[4pt]
& x_{i,c,w,\tau}\in\{0,1\}
&& \forall\, i\in\mathcal{F},\, \tau\in\mathcal{S}_i,\, W^{\min}_{i,r_c}\le w\le W^{\max}_{i,r_c} \tag{8}\\[4pt]
& y_{c,k,m}\in\{0,1\}
&& \forall\, c\in\mathcal{C},\, 1\le k\le K,\, 1\le m\le M \tag{9}
\end{align}$$
