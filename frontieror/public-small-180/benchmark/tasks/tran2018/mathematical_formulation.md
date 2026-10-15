# Sets and Indices

- $N = \{1,2,\ldots,n\}$: set of nodes (candidate station locations), indexed by $i$.

- $Q$: set of paths (origin-destination pairs), indexed by $q$.

- $\vec{a}_{jk}$: directed arc from node $j$ to node $k$.

- $\vec{A}_q$: set of directed arcs traversed on path $q$, sorted from origin to destination and back (round trip).

- $K^q_{jk}$: set of candidate nodes whose refueling allows a vehicle (with range $R$) to traverse arc $\vec{a}_{jk} \in \vec{A}_q$ without running out of fuel.

# Parameters

- $f_q$: volume of traffic flow on path $q \in Q$.

- $p$: number of stations to be located.

- $R$: driving range of the vehicles.

# Decision Variables

- $x_i \in \{0,1\}$: equals $1$ if a station is located at node $i \in N$, $0$ otherwise.

- $y_q \in \{0,1\}$: equals $1$ if the flow on path $q \in Q$ is refueled, $0$ otherwise.

# Formulation \[ASLP\]

$$\begin{align}
  \max \quad z_0 &= \sum_{q \in Q} f_q\, y_q \tag{14} \\
  \text{s.t.}\quad
  \sum_{i \in K^q_{jk}} x_i &\ge y_q, && \forall\, q \in Q,\ \vec{a}_{jk} \in \vec{A}_q \tag{15} \\
  \sum_{i \in N} x_i &= p \tag{16} \\
  x_i,\ y_q &\in \{0,1\}, && \forall\, i \in N,\ q \in Q \tag{17}
\end{align}$$

Constraints (15) are the arc-coverage conditions: a path $q$ may be counted as refueled only if, for every directed arc on its round trip, at least one station in the corresponding candidate set is open. Constraint (16) forces exactly $p$ stations to be opened, and (17) enforces binarity.
