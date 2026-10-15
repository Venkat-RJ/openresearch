# Original Formulation: VRP with Divisible Deliveries and Pickups (as VRPMDP)

**Source.** Nagy, Wassan, Speranza, Archetti (2015), *Transportation Science* 49(2):271–294. Section 5.1 modelling of the VRPDDP as a VRP with Mixed Deliveries and Pickups (VRPMDP) by splitting each original customer into a co-located linehaul/backhaul pair.

## Sets and Indices

$$\begin{align*}
&\mathbf{D} = \{0\} && \text{single depot} \\
&\mathbf{L} = \{1,\ldots,n\} && \text{linehaul (delivery) customers} \\
&\mathbf{B} = \{n+1,\ldots,2n\} && \text{backhaul (pickup) customers; $n+i$ is co-located with $i\in\mathbf{L}$} \\
&\mathbf{V} = \mathbf{D}\cup\mathbf{L}\cup\mathbf{B} && \text{all vertices}
\end{align*}$$

## Parameters

$$\begin{align*}
& d_{ij} && \text{distance between locations $i$ and $j$, $i,j \in \mathbf{V}$} \\
& q_i && \text{demand of customer $i$ (delivery if $i\in\mathbf{L}$, pickup if $i\in\mathbf{B}$)} \\
& C && \text{vehicle capacity}
\end{align*}$$

## Decision Variables

$$\begin{align*}
& x_{ij} \in \{0,1\} && \text{$=1$ if a vehicle travels from $i$ to $j$} \\
& R_{ij} \ge 0 && \text{delivery load carried on arc $(i,j)$} \\
& P_{ij} \ge 0 && \text{pickup load carried on arc $(i,j)$}
\end{align*}$$

## Formulation

$$\begin{align}
\min\ Z\ =\ & \sum_{i\in\mathbf{V}} \sum_{j\in\mathbf{V}} d_{ij}\,x_{ij} \tag{1}\\
\text{s.t.}\
& \sum_{i\in\mathbf{V}} x_{ij} \;=\; 1
   && \forall\, j \in \mathbf{L}\cup\mathbf{B} \tag{2}\\
& \sum_{i\in\mathbf{V}} x_{ji} \;=\; 1
   && \forall\, j \in \mathbf{L}\cup\mathbf{B} \tag{3}\\
& \sum_{i\in\mathbf{V}} R_{ij}\;-\;q_j \;=\; \sum_{i\in\mathbf{V}} R_{ji}
   && \forall\, j \in \mathbf{L} \tag{4}\\
& \sum_{i\in\mathbf{V}} R_{ij} \;=\; \sum_{i\in\mathbf{V}} R_{ji}
   && \forall\, j \in \mathbf{B} \tag{5}\\
& \sum_{i\in\mathbf{V}} P_{ij} \;=\; \sum_{i\in\mathbf{V}} P_{ji}
   && \forall\, j \in \mathbf{L} \tag{6}\\
& \sum_{i\in\mathbf{V}} P_{ij}\;+\;q_j \;=\; \sum_{i\in\mathbf{V}} P_{ji}
   && \forall\, j \in \mathbf{B} \tag{7}\\
& \sum_{i\in\mathbf{L}\cup\mathbf{B}} P_{0i} \;=\; 0 \tag{8}\\
& \sum_{i\in\mathbf{L}\cup\mathbf{B}} R_{i0} \;=\; 0 \tag{9}\\
& R_{ij} + P_{ij} \;\le\; C\,x_{ij}
   && \forall\, i,j \in \mathbf{V} \tag{10}\\
& x_{ij} \in \{0,1\}
   && \forall\, i,j \in \mathbf{V} \tag{11}\\
& R_{ij} \ge 0
   && \forall\, i,j \in \mathbf{V} \tag{12}\\
& P_{ij} \ge 0
   && \forall\, i,j \in \mathbf{V} \tag{13}
\end{align}$$

\(1\) minimizes total travel distance. (2)–(3) enforce exactly one predecessor/successor for each customer. (4)–(7) are flow conservation of delivery and pickup loads at linehaul and backhaul customers. (8)–(9) initialize vehicles with zero pickup load leaving the depot and require zero delivery load returning to the depot. (10) enforces vehicle capacity on every arc used. (11)–(13) are variable domains.
