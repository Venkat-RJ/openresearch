# Original Formulation: Truck-and-Trailer Routing Problem with Time Windows (TTRPTW)

*Source: Branch-and-Price-and-Cut for the Truck-and-Trailer Routing Problem with Time Windows, Rothenbächer, Drexl, and Irnich, 2018 (Transportation Science).*

This is the path-based (set-partitioning) model presented in Section 3.1, the formulation whose notation is carried through every subsequent section of the paper.

## Sets and Indices

$$\begin{align*}
N        &: \text{set of customers} \\
N^L      &\subset N: \text{set of truck customers (accessible only by a truck without its trailer)} \\
N^H      &\subset N: \text{set of trailer customers (accessible by a single truck or a complete vehicle)} \\
L        &: \text{set of truck types} \\
H        &: \text{set of trailer types} \\
K        &\subseteq L \times (H \cup \{0\}): \text{set of vehicle classes;}\\
         &\quad k=(l,0)\ \text{is a single truck of type }l,\quad k=(l,h),\,h\neq 0,\ \text{is a complete vehicle} \\
f        &\in L \cup H: \text{a vehicle type index} \\
R^k      &: \text{set of all valid routes for vehicle class } k \in K
\end{align*}$$

## Parameters

$$\begin{align*}
c_r      &: \text{cost of route } r \text{ (variable travel costs, depending on distance and on}\\
         &\quad \text{whether the truck tows the trailer, plus coupling/decoupling/load-transfer handling costs)} \\
a_{rn}   &: \text{number of times route } r \text{ serves customer } n \text{ (visits used only for parking,}\\
         &\quad \text{coupling, or load transfer are not counted)} \\
b^k_f    &\in\{0,1\}: \text{vehicle indicator, } 1 \text{ if vehicle class } k \in K \text{ uses vehicle type } f \in L\cup H \\
n_f      &: \text{number of available vehicles of type } f \in L \cup H
\end{align*}$$

## Decision Variables

$$\begin{align*}
\lambda^k_r \in \mathbb{Z}_{\geq 0}
   &: \text{number of times route } r \text{ of vehicle class } k \text{ is chosen,}\\
   &\quad \forall\, k \in K,\; r \in R^k.
\end{align*}$$

## Objective

$$\begin{align}
\min \quad & \sum_{k \in K} \sum_{r \in R^k} c_r \, \lambda^k_r \tag{1a}
\end{align}$$

## Constraints

$$\begin{align}
\text{s.t.}\quad
& \sum_{k \in K} \sum_{r \in R^k} a_{rn}\, \lambda^k_r = 1
   && (\pi_n) && \forall\, n \in N, \tag{1b}\\[2pt]
& \sum_{k \in K} \sum_{r \in R^k} b^k_f\, \lambda^k_r \leq n_f
   && (\mu_f) && \forall\, f \in L \cup H, \tag{1c}\\[2pt]
& \lambda^k_r \geq 0,\ \text{integer}
   && && \forall\, k \in K,\; r \in R^k. \tag{1d}
\end{align}$$
