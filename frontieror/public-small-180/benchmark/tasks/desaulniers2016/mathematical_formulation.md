# Original Formulation: Inventory-Routing Problem (IRP)

*Source: A Branch-Price-and-Cut Algorithm for the Inventory-Routing Problem, Guy Desaulniers, Jørgen G. Rakke, Leandro C. Coelho, 2016.*

This is the innovative mixed-integer program (1)–(9) stated in §2 “Problem Definition and Mathematical Model.” It is a route / route-delivery-pattern (RDP) formulation under the maximum-level (ML) replenishment policy. The route set $R$ and the extreme-RDP sets $W_r^p$ have exponential cardinality (the paper solves it by branch-price-and-cut); per the formulation’s definition these index sets are kept as-is and are *not* replaced by an arc-flow reformulation.

## Sets and Indices

$$\begin{align*}
& 0 : \text{the single supplier (depot)} \\
& N : \text{set of customers; each } i \in N \text{ consumes } d_i^p \text{ in period } p \\
& P = \{1, 2, \ldots, \rho\} : \text{finite planning horizon; } \rho+1 \text{ is a fictitious period for end inventories} \\
& R : \text{set of feasible routes (each starts and ends at the depot } 0\text{)} \\
& N_r : \text{set of customers visited in route } r \in R \\
& A_r : \text{set of pairs of locations } (i,j) \text{ visited consecutively along route } r \\
& W_r^p : \text{set of extreme RDPs available when route } r \text{ is used in period } p \in P \\
& P_{ip}^+ \subseteq \{p, p{+}1, \ldots, \rho{+}1\} : \text{set of periods to which a subdelivery of a delivery} \\
&\qquad\quad \text{to customer } i \text{ in period } p \text{ can be dedicated (assumed nonempty)} \\
& P_{is}^- = \{p \in P \mid s \in P_{ip}^+\} : \text{set of periods at which a subdelivery can be made to} \\
&\qquad\quad \text{fulfill the demand of customer } i \text{ at period } s
\end{align*}$$

## Parameters

$$\begin{align*}
& K : \text{number of vehicles (homogeneous fleet)} \\
& Q : \text{vehicle capacity} \\
& d_0^p : \text{quantity produced at the supplier in period } p \in P \\
& d_i^p : \text{demand (consumption) of customer } i \in N \text{ in period } p \in P \\
& C_i\ (C_0) : \text{inventory capacity of customer } i \in N \text{ (supplier } 0\text{)} \\
& I_i^0 \le C_i\ (I_0^0 \le C_0) : \text{initial inventory of customer } i \in N \text{ (supplier } 0\text{)} \\
& h_i\ (h_0) : \text{unit holding cost at customer } i \in N \text{ (supplier } 0\text{)} \\
& c_{ij} : \text{traveling cost between locations } i \text{ and } j \\
& a_{ri} \in \{0,1\} : \text{1 if route } r \text{ visits customer } i \in N, \text{ 0 otherwise} \\
& I_i^{0,s} = \max\!\Big\{0,\ I_i^0 - \textstyle\sum_{\ell=1}^{s} d_i^\ell\Big\} : \text{initial inventory remaining at customer } i \text{ at end of period } s \\
& q_{wi}^s \in [0,\ u_{ip}^s] : \text{quantity in extreme RDP } w \in W_r^p \text{ delivered to customer } i \in N_r \text{ and} \\
&\qquad\quad \text{dedicated to period } s \in P_{ip}^+ \\
& q_w = \textstyle\sum_{i \in N_r}\sum_{s \in P_{ip}^+} q_{wi}^s : \text{total quantity delivered in RDP } w \text{ (loaded at the supplier)} \\
& b_{wi}^s : \text{quantity delivered to customer } i \text{ in RDP } w \text{ that remains in inventory at end of period } s \\
& c_{rw} = \textstyle\sum_{(i,j) \in A_r} c_{ij} + \sum_{i \in N_r}\sum_{s \in P_{ip}^+} h_i\, b_{wi}^s : \text{cost associated with route } r \text{ and RDP } w
\end{align*}$$

Residual demands and subdelivery upper bounds: $$\begin{align*}
& \bar{d}_i^s =
  \begin{cases}
    \max\{0,\ d_i^1 - I_i^0\} & \text{if } s = 1 \\
    \max\{0,\ d_i^s - I_i^{0,s-1}\} & \text{otherwise}
  \end{cases}
  \qquad \forall\, s \in P, \\[4pt]
& u_{ip}^s =
  \begin{cases}
    \min\{\bar{d}_i^s,\ C_i - I_i^{0,s-1}\} & \text{if } s = p \\
    C_i - \sum_{\ell=p}^{s-1} d_i^\ell - I_i^{0,s-1} & \text{if } s = \rho+1 \\
    \min\{\bar{d}_i^s,\ C_i - \sum_{\ell=p}^{s-1} d_i^\ell - I_i^{0,s-1}\} & \text{otherwise.}
  \end{cases}
\end{align*}$$

## Decision Variables

$$\begin{align*}
& y_{rw}^p \in [0,1] : \text{proportion of route } r \in R \text{ operated with RDP } w \in W_r^p \text{ in period } p \in P \\
& I_0^p \ge 0 : \text{inventory at the supplier at the end of period } p \in P
\end{align*}$$

## Objective

$$\begin{equation}
\min\ \left\{ \sum_{p \in P} \sum_{r \in R} \sum_{w \in W_r^p} c_{rw}\, y_{rw}^p
  + \sum_{p \in P} h_0\, I_0^p \right\} \tag{1}
\end{equation}$$

## Constraints

$$\begin{align}
& I_0^{p-1} + d_0^p - \sum_{r \in R} \sum_{w \in W_r^p} q_w\, y_{rw}^p = I_0^p,
  && \forall\, p \in P, \tag{2} \\[4pt]
& \sum_{p \in P_{is}^-} \sum_{r \in R} \sum_{w \in W_r^p} q_{wi}^s\, y_{rw}^p = \bar{d}_i^s,
  && \forall\, i \in N,\ s \in P \text{ s.t. } \bar{d}_i^s > 0, \tag{3} \\[4pt]
& I_i^{0,s} + \sum_{p \in P_{is}^-} \sum_{r \in R} \sum_{w \in W_r^p} b_{wi}^s\, y_{rw}^p + d_i^s \le C_i,
  && \forall\, i \in N,\ s \in P, \tag{4} \\[4pt]
& \sum_{r \in R} \sum_{w \in W_r^p} a_{ri}\, y_{rw}^p \le 1,
  && \forall\, i \in N,\ p \in P, \tag{5} \\[4pt]
& \sum_{r \in R} \sum_{w \in W_r^p} y_{rw}^p \le K,
  && \forall\, p \in P, \tag{6} \\[4pt]
& 0 \le I_0^p \le C_0,
  && \forall\, p \in P, \tag{7} \\[4pt]
& y_{rw}^p \ge 0,
  && \forall\, p \in P,\ r \in R,\ w \in W_r^p, \tag{8} \\[4pt]
& \sum_{w \in W_r^p} y_{rw}^p \in \{0,1\},
  && \forall\, p \in P,\ r \in R. \tag{9}
\end{align}$$
