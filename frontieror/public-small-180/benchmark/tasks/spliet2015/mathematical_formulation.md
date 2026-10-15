# Original Formulation: Time Window Assignment Vehicle Routing Problem (TWAVRP)

*Source: The Time Window Assignment Vehicle Routing Problem, Remy Spliet, Adriana F. Gabor, 2014 (Transportation Science, Articles in Advance).*

## Sets and Indices

$$\begin{align*}
G = (V, A) &: \text{complete graph}. \\
V = \{0, 1, \dots, n+1\} &: \text{set of locations; } 0 \text{ is the starting depot, } n+1 \text{ the ending depot}. \\
V' = \{1, \dots, n\} &: \text{set of customers}. \\
A &: \text{set of arcs of } G. \\
\Omega &: \text{set of scenarios, each representing a realization of demand}. \\
R(\omega) &: \text{set of all feasible routes for scenario } \omega \in \Omega. \\
(i,j) &: \text{arc index, } (i,j) \in A. \\
v &: \text{location/customer index.} \\
r &: \text{route index.} \\
\omega &: \text{scenario index.}
\end{align*}$$

## Parameters

$$\begin{align*}
c_{ij} \geq 0 &: \text{travel cost along arc } (i,j) \text{ (satisfies the triangle inequality)}. \\
t_{ij} \geq 0 &: \text{travel time along arc } (i,j) \text{ (satisfies the triangle inequality)}. \\
Q &: \text{capacity of each vehicle (unlimited number of identical vehicles)}. \\
p_\omega &: \text{probability that scenario } \omega \text{ occurs}. \\
d_v^\omega &: \text{demand at location } v \text{ in scenario } \omega, \text{ with } 0 < d_v^\omega \leq Q \text{ for } v \in V', \\
          &\quad \text{ and } d_0^\omega = d_{n+1}^\omega = 0. \\
[s_v, e_v] &: \text{exogenous time window at location } v \in V. \\
w_v &: \text{prescribed width of the endogenous time window of customer } v \in V', \\
    &\quad \text{ assumed } s_v \leq e_v - w_v. \\
a_r^v &: \text{number of times customer } v \in V' \text{ is visited by route } r. \\
t_r^v &: \text{cumulative time of service of customer } v \in V' \text{ on route } r \\
      &\quad (0 \text{ if not visited; the service time if visited once; the sum if visited multiple times)}. \\
c_r = \sum_{i=1}^{k} c_{r_i} &: \text{cost of route } r \text{ with arcs } \{r_1, \dots, r_k\}.
\end{align*}$$

## Decision Variables

$$\begin{align*}
x_r^\omega &\in \{0,1\} : \text{equals } 1 \text{ if route } r \text{ is used in scenario } \omega, \text{ and } 0 \text{ otherwise.} \\
y_v &\in [s_v,\, e_v - w_v] : \text{start time of the endogenous time window assigned to customer } v \in V' \\
    &\quad \text{(assigned before demand is revealed; identical across all scenarios).}
\end{align*}$$

## Objective

$$\begin{align}
\min \;\; \sum_{\omega \in \Omega} p_\omega \sum_{r \in R(\omega)} c_r\, x_r^\omega \tag{1}
\end{align}$$

## Constraints

$$\begin{align}
\sum_{r \in R(\omega)} a_r^v\, x_r^\omega &= 1
    && \forall v \in V', \; \forall \omega \in \Omega, \tag{2} \\[4pt]
\sum_{r \in R(\omega)} t_r^v\, x_r^\omega &\geq y_v
    && \forall v \in V', \; \forall \omega \in \Omega, \tag{3} \\[4pt]
\sum_{r \in R(\omega)} t_r^v\, x_r^\omega &\leq y_v + w_v
    && \forall v \in V', \; \forall \omega \in \Omega, \tag{4} \\[4pt]
x_r^\omega &\in \{0,1\}
    && \forall \omega \in \Omega, \; \forall r \in R(\omega), \tag{5} \\[4pt]
y_v &\in [s_v,\, e_v - w_v]
    && \forall v \in V'. \tag{6}
\end{align}$$
