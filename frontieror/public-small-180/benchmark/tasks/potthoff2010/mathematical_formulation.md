# Original Formulation: Operational Crew Rescheduling Problem (OCRSP)

*Source: Column Generation with Dynamic Duty Selection for Railway Crew Rescheduling, M. Potthoff, D. Huisman, and G. Desaulniers, Transportation Science 44(4), pp. 493–505, 2010.*

## Sets and Parameters

$$\begin{align*}
S        &: \text{set of stations (limited to relief points).}\\
D        &: \text{set of crew bases (a subset of the relief points).}\\
N        &: \text{set of tasks that have not started at the time of rescheduling.}\\
         &\quad \text{For every } i\in N:\ ds_i,dt_i \text{ are its departure station and time,}\\
         &\quad \text{and } as_i,at_i \text{ are its arrival station and time.}\\
\Delta   &= \Delta_A \cup \Delta_R:\ \text{set of unfinished original duties, where } \Delta_A\\
         &\quad \text{are the active and } \Delta_R \text{ the reserve duties. For every } \delta\in\Delta:\\
         &\quad cs_\delta:\ \text{the station where the duty is at the time of rescheduling}\\
         &\quad\quad \text{(or the arrival station of the task being performed then);}\\
         &\quad b_\delta:\ \text{the crew base where the original duty starts and ends.}\\
K^\delta &: \text{set of all feasible completions for original duty } \delta\in\Delta.\\
         &\quad \text{For every feasible completion } k\in K^\delta:\\
         &\quad c^\delta_k:\ \text{cost of feasible completion } k \text{ for duty } \delta \text{ (zero if the}\\
         &\quad\quad \text{duty is unchanged);}\\
         &\quad a^\delta_{ik}\in\{0,1\}:\ 1 \text{ if task } i \text{ is covered (driven) by completion } k.\\
f_i      &: \text{cost for canceling task } i.
\end{align*}$$

## Decision Variables

$$\begin{align*}
x^\delta_k &\in \{0,1\} \quad \forall\, \delta\in\Delta,\ \forall\, k\in K^\delta,\\
           &\quad =1 \text{ if feasible completion } k \text{ is selected for original duty } \delta.\\
y_i        &\in \{0,1\} \quad \forall\, i\in N,\\
           &\quad =1 \text{ if task } i \text{ is canceled.}
\end{align*}$$

## Objective

$$\begin{align}
\min \quad \sum_{\delta\in\Delta}\sum_{k\in K^\delta} c^\delta_k\, x^\delta_k
         + \sum_{i\in N} f_i\, y_i \tag{1}
\end{align}$$

## Constraints

$$\begin{align}
\sum_{\delta\in\Delta}\sum_{k\in K^\delta} a^\delta_{ik}\, x^\delta_k + y_i &\ge 1
   && \forall\, i\in N \tag{2}\\
\sum_{k\in K^\delta} x^\delta_k &= 1
   && \forall\, \delta\in\Delta \tag{3}\\
x^\delta_k,\ y_i &\in \{0,1\}
   && \forall\, \delta\in\Delta,\ \forall\, k\in K^\delta,\ \forall\, i\in N \tag{4}
\end{align}$$

Constraints (2) are set-covering constraints ensuring that every task is either covered by a selected feasible completion or canceled (over-covering is allowed, hence “$\ge$”). Constraints (3) ensure that every original duty is assigned to exactly one feasible completion.

## Released finite-completion contract

The public data use `released_problem_variant = relief_point_completion_v1`.
The executable solver constructs a deterministic polynomial-size subset of
$K^\delta$ consisting of the unchanged/empty completion and individually
reachable task completions, then solves (1)--(4) over that finite library.
Each emitted completion contains its ordered tasks, exact taxi flag, terminal
station/time, and an explicit emergency relief-point-release flag. The latter
is permitted only when the frozen duty horizon has no timely base-return
connection; it exposes a fallback that the former generator used implicitly.

For this released library, $c_k^\delta$ is recomputed as the fixed duty-change
cost (unless unchanged), plus per-reassigned-task and per-new-transfer costs,
plus the taxi cost when used. No coefficient is fabricated for meal breaks,
short connections, or overtime because those coefficients are absent from the
instance. The checker independently validates the ordered completion and
recomputes both $c_k^\delta$ and objective (1).
