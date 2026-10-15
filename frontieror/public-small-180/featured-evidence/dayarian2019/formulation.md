# Original Formulation: Caterer Production-Routing Problem (CPRP)

*Source: A Branch-Price-and-Cut Algorithm for a Production-Routing Problem with Short-Life-Span Products, I. Dayarian and G. Desaulniers, Transportation Science (Articles in Advance, pp. 1–21), 2019.*

## Sets and Indices

- $\mathcal{T}$ : set of time periods, numbered chronologically $1,\dots,|\mathcal{T}|$; period $t$ has start time $b^t$ and end time $e^t$, with $e^t=b^{t+1}$ for $t\in\mathcal{T}\setminus\{|\mathcal{T}|\}$.

- $\mathcal{U}$ : set of subperiods, numbered $1,\dots,|\mathcal{U}|$; subperiod $u$ has start time $b^u$ and end time $e^u$.

- $\mathcal{P}$ : set of products.

- $\mathcal{O}$ : set of orders (customers), indexed by $i$.

- $\mathcal{P}_i\subseteq\mathcal{P}$ : products with positive demand for order $i\in\mathcal{O}$.

- $\mathcal{U}_i^p$ : subperiods during which demand of product $p$ for order $i$ may be produced (set by release time and minimum travel time to $i$).

- $\mathcal{U}^p=\bigcup_{i\in\mathcal{O}}\mathcal{U}_i^p$ : subperiods in which product $p$ may be produced for some order.

- $\mathcal{O}_u^p$ : orders $i$ such that $u\in\mathcal{U}_i^p$.

- $\mathcal{T}_i\subseteq\mathcal{T}$ : periods at the beginning of which a trip may start to deliver order $i$ within its time window.

- $\mathcal{R}$ : set of all feasible delivery routes, indexed by $r$.

- $\mathcal{S}$ : set of all feasible production employee shifts, indexed by $s$ (a shift also describes the employee assignment in each of its subperiods).

## Parameters

- $\eta^p$ : production rate of product $p$ (units per subperiod).

- $d_i^p$ : demand of order $i$ for product $p$.

- $n^E,\ n^W$ : maximum number of production employees, maximum number of workstations.

- $n^V,\ n^D$ : number of available vehicles, number of available drivers.

- For route $r\in\mathcal{R}$: cost $c_r$ (driver salary $+$ vehicle operational cost); $a_{ir}\in\{0,1\}$, $1$ if $r$ visits customer $i$; $g_{ir}^t\in\{0,1\}$, $1$ if $i$ is visited in $r$ on a trip starting at $b^t$ (for $i\in\mathcal{O}$, $t\in\mathcal{T}_i$); $b_r^t\in\{0,1\}$, $1$ if $r$ starts at $b^t$; $e_r^t\in\{0,1\}$, $1$ if $r$ ends at $e^t$.

- For shift $s\in\mathcal{S}$: cost $c_s$ (employee salary $+$ setup costs); $f_s^{pu}\in\{0,1\}$, $1$ if $s$ is assigned to product $p$ in subperiod $u$ (for $p\in\mathcal{P}$, $u\in\mathcal{U}^p$); $b_s^t\in\{0,1\}$, $1$ if $s$ starts at $b^t$; $e_s^t\in\{0,1\}$, $1$ if $s$ ends at $e^t$.

## Decision Variables

$$\begin{align*}
  \theta_r   &\in\{0,1\}, && r\in\mathcal{R}: \ 1 \text{ if route } r \text{ is selected.}\\
  \psi_s     &\in\mathbb{Z}_{\ge 0}, && s\in\mathcal{S}: \text{ number of production employees working on shift } s.\\
  q_i^{pu}   &\in\mathbb{R}_{\ge 0}, && i\in\mathcal{O},\,p\in\mathcal{P}_i,\,u\in\mathcal{U}_i^p: \text{ units of product } p \text{ produced in subperiod } u \text{ for order } i.\\
  v^t        &\in\mathbb{Z}, && t\in\mathcal{T}: \text{ number of vehicles used in period } t.\\
  w^t        &\in\mathbb{Z}, && t\in\mathcal{T}: \text{ number of workstations used in period } t.
\end{align*}$$

## Objective

$$\begin{equation}
  \min \quad \sum_{r\in\mathcal{R}} c_r\,\theta_r \;+\; \sum_{s\in\mathcal{S}} c_s\,\psi_s \tag{1}
\end{equation}$$

## Constraints

$$\begin{align}
  \sum_{r\in\mathcal{R}} a_{ir}\,\theta_r &= 1, & \forall i\in\mathcal{O}, \tag{2}\\[4pt]
  \sum_{i\in\mathcal{O}_u^p} q_i^{pu} &\le \eta^p \sum_{s\in\mathcal{S}} f_s^{pu}\,\psi_s, & \forall p\in\mathcal{P},\ u\in\mathcal{U}^p, \tag{3}\\[4pt]
  \sum_{\substack{u\in\mathcal{U}_i^p\\ e^u\le b^t}} q_i^{pu} &\ge \sum_{r\in\mathcal{R}} d_i^p\, g_{ir}^t\,\theta_r, & \forall i\in\mathcal{O},\ p\in\mathcal{P}_i,\ t\in\mathcal{T}_i, \tag{4}\\[4pt]
  \sum_{r\in\mathcal{R}} \theta_r &\le n^D, \tag{5}\\[4pt]
  \sum_{s\in\mathcal{S}} \psi_s &\le n^E, \tag{6}\\[4pt]
  v^{t-1} + \sum_{r\in\mathcal{R}} \bigl(b_r^t - e_r^{t-1}\bigr)\theta_r &= v^t, & \forall t\in\mathcal{T}, \tag{7}\\[4pt]
  w^{t-1} + \sum_{s\in\mathcal{S}} \bigl(b_s^t - e_s^{t-1}\bigr)\psi_s &= w^t, & \forall t\in\mathcal{T}, \tag{8}\\[4pt]
  \theta_r &\in \{0,1\}, & \forall r\in\mathcal{R}, \tag{9}\\[4pt]
  \psi_s &\ge 0,\ \text{integer}, & \forall s\in\mathcal{S}, \tag{10}\\[4pt]
  q_i^{pu} &\ge 0, & \forall i\in\mathcal{O},\ p\in\mathcal{P}_i,\ u\in\mathcal{U}_i^p, \tag{11}\\[4pt]
  0 \le v^t &\le n^V,\ \text{integer}, & \forall t\in\mathcal{T}, \tag{12}\\[4pt]
  0 \le w^t &\le n^W,\ \text{integer}, & \forall t\in\mathcal{T}. \tag{13}
\end{align}$$

In constraints (7) and (8), for $t=1$ we set $v^0=w^0=e_r^0=e_s^0=0$ for all $r\in\mathcal{R}$ and $s\in\mathcal{S}$.

## Released Compact Benchmark Contract

The executable benchmark uses a finite, aggregate formulation in place of
enumerating $\mathcal R$ and $\mathcal S$.  This section is normative for the
released instances, solver, solution schema, and checker.

For each vehicle $v$ and trip slot $k$, the model selects an active prefix of
trip slots, a start period, an ordered depot-to-depot customer path, and its
customer assignments.  Consecutive active slots must have strictly increasing
start-period indices.  The compact contract deliberately does **not** add a
continuous-time constraint requiring trip $k$ to return before trip $k+1$
departs.  An inactive slot has start time zero.  Arrival times follow the
reported path, preparation, travel, waiting, and service times.  The route-end
variable is bounded below by every served customer's arrival plus service and
direct return-to-depot time.  Driver paid time is the maximum of this modeled
route duration and the guaranteed minimum; vehicle cost is based on the exact
arcs in the reported order.

The production completion is represented by nonnegative integers
$f^{pu}$ (producing employees), $\beta^{pu}$ (setup starts), $w^t$ (paid
workstations), and $\alpha^t$ (employee-shift starts).  Besides (3), the
released model enforces

$$
f^{pu} \le f^{p,u-1}+\beta^{p,u-\sigma},
$$

with absent predecessor terms equal to zero.  In each subperiod $u$ of period
$t$, $w^t$ covers all producing employees plus setup employees whose
$\sigma$-subperiod setup interval contains $u$; the same count cannot exceed
$n^E$.  Shift starts satisfy

$$
\alpha^0\ge w^0,\qquad
\alpha^t\ge w^t-w^{t-1},\qquad
\sum_t\alpha^t\le n^E,
$$

and $w^t$ covers the preceding shift starts over the configured minimum paid
periods.  All four arrays are integral and use the instance employee and
workstation bounds.

The JSON certificate stores the selected customer order and every
$f,\beta,w,\alpha$ value.  Its objective is checked by exact recomputation:

$$
\gamma_D\sum_v \operatorname{paid}_v
+\gamma_V\sum_v \operatorname{travel}_v
+\gamma_E n^{SP}\sum_t w^t
+\gamma_S\sum_{p,u}\beta^{pu}.
$$
