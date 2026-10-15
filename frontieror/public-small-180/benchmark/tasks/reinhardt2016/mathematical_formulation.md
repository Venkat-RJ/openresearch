# Original Formulation: Edge Set Cost Vehicle Routing Problem with Time Windows (ESVRPTW)

*Source: The Edge Set Cost of the Vehicle Routing Problem with Time Windows, Line Blander Reinhardt, Mads Kehlet Jepsen, David Pisinger, Transportation Science 50(2), 2016.*

## Sets and Indices

$$\begin{align*}
R   &: \text{the set of all edge sets} \\
C   &: \text{the set of customers; } C_r \text{ are the customers covered by the edges in set } r \in R \\
V   &: \text{the set of nodes representing the customers in } C \text{ and the depot, defined as } 0 \;\; (V = \{0\} \cup C) \\
\mathbf{A} &: \text{the set of arcs } (i,j) \text{ in } V; \;\; \mathbf{A}_r \text{ is the set of arcs } (i,j) \text{ belonging to set } r \in R \\
K   &: \text{the set of vehicles, where } |K| \leq |C| \text{ as usual for VRPTW problems}
\end{align*}$$

## Parameters

$$\begin{align*}
D          &: \text{the capacity of the vehicles} \\
d_i        &: \text{the demand which must be delivered to node } i \in V \text{ (the demand at the depot is zero)} \\
a_i        &: \text{the availability (earliest service) time for customer } i \in C, \;\; a_i \geq 0 \\
b_i        &: \text{the required completion (latest service) time for customer } i \in C, \;\; b_i \geq a_i \\
c_{ij}     &: \text{the cost of using arc } (i,j) \in \mathbf{A} \\
\theta_{ij}&: \text{the time it takes to travel on arc } (i,j) \in \mathbf{A} \\
c_r        &: \text{the cost of accessing the arcs in edge set } r \in R \\
M          &: \text{a large constant greater than the duration of any route}
\end{align*}$$

## Decision Variables

$$\begin{align*}
x_{ij}^{v} &\in \{0,1\} : \text{one if arc } (i,j) \in \mathbf{A} \text{ is used by vehicle } v \in K, \text{ zero otherwise} \\
y_r        &\in \{0,1\} : \text{one if an edge from edge set } r \in R \text{ is used, zero otherwise} \\
t_i^{v}    &\in \mathbb{R}_0^{+} : \text{the time vehicle } v \in K \text{ visits node } i \in V
\end{align*}$$

## Objective

$$\begin{align}
\textbf{DPM:} \quad \min \quad & \sum_{v \in K} \sum_{(i,j) \in \mathbf{A}} c_{ij}\, x_{ij}^{v} \;+\; \sum_{r \in R} c_r\, y_r \tag{15}
\end{align}$$

## Constraints

$$\begin{align}
\text{s.t.} \quad
& \sum_{v \in K} \sum_{j:(j,i) \in \mathbf{A}_r} x_{ji}^{v} \leq y_r & \forall r \in R,\; i \in C_r \tag{16} \\
& \sum_{v \in K} \sum_{j:(i,j) \in \mathbf{A}_r} x_{ij}^{v} \leq y_r & \forall r \in R,\; i \in C_r \tag{17} \\
& \sum_{v \in K} \sum_{j:(i,j) \in \mathbf{A}} x_{ij}^{v} = 1 & \forall i \in C \tag{18} \\
& \sum_{i \in C} x_{i0}^{v} = \sum_{i \in C} x_{0i}^{v} & \forall v \in K \tag{19} \\
& \sum_{j:(j,i) \in \mathbf{A}} x_{ji}^{v} - \sum_{j:(i,j) \in \mathbf{A}} x_{ij}^{v} = 0 & \forall i \in C,\; \forall v \in K \tag{20} \\
& \sum_{(i,j) \in \mathbf{A}} d_i\, x_{ij}^{v} \leq D & \forall v \in K \tag{21} \\
& a_i \leq t_i^{v} \leq b_i & \forall i \in V,\; v \in K \tag{22} \\
& (t_i^{v} + \theta_{ij})\, x_{ij}^{v} - t_j^{v} \leq 0 & \forall v \in K,\; (i,j) \in \mathbf{A} \tag{23} \\
& x_{ij}^{v} \in \{0,1\} & \forall (i,j) \in \mathbf{A},\; v \in K \tag{24} \\
& y_r \in \{0,1\} & \forall r \in R \tag{25} \\
& t_i^{v} \in \mathbb{R}_0^{+} & \forall i \in V,\; v \in K \tag{26}
\end{align}$$

The nonlinear time-propagation constraint (23) can be replaced by the linear big-$M$ constraints: $$\begin{align}
& t_i^{v} + \theta_{ij} - t_j^{v} \leq M\,(1 - x_{ij}^{v}) & \forall v \in K,\; (i,j) \in \mathbf{A} \tag{27}
\end{align}$$ where $M$ is a large constant greater than the duration of any route.
