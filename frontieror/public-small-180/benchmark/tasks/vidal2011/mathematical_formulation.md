# Original Formulation: Multi-Depot Periodic Vehicle Routing Problem (MDPVRP)

*Source: A Hybrid Genetic Algorithm for Multi-Depot and Periodic Vehicle Routing Problems, Thibaut Vidal, Teodor Gabriel Crainic, Michel Gendreau, Nadia Lahrichi, Walter Rei, 2011 (CIRRELT-2011-05). Five-index vehicle flow formulation, Annex 7.1.*

## Sets and Indices

$$\begin{align*}
G = (V, A) \;:\;& \text{complete graph}. \\
V = V^{\mathrm{DEP}} \cup V^{\mathrm{CST}} \;:\;& \text{complete vertex set}. \\
V^{\mathrm{DEP}} = \{v_0, \ldots, v_d\} \;:\;& \text{set of depot vertices}. \\
V^{\mathrm{CST}} = \{v_1, \ldots, v_n\} \;:\;& \text{set of customer vertices},\ i = 1, \ldots, n. \\
A = \{a_{ij} : i,j \in V\} \;:\;& \text{arc set}. \\
L_i \;:\;& \text{list of feasible visit combinations (patterns) for customer } i. \\
v_o \in V^{\mathrm{DEP}} \;:\;& \text{depot index}. \\
p \in L_i \;:\;& \text{visit-combination (pattern) index for customer } i. \\
k = 1, \ldots, m \;:\;& \text{vehicle index ($m$ vehicles available at each depot)}. \\
l = 1, \ldots, t \;:\;& \text{period (day) index ($t$ periods)}. \\
S \subseteq V^{\mathrm{CST}},\ |S| \geq 2 \;:\;& \text{customer subsets (for subtour elimination)}.
\end{align*}$$

## Parameters

$$\begin{align*}
c_{ij} \;:\;& \text{routing cost (travel time) from vertex } v_i \text{ to vertex } v_j. \\
q_i \;:\;& \text{non-negative demand of customer } i. \\
\tau_i \;:\;& \text{service duration at customer } i. \\
Q \;:\;& \text{vehicle capacity}. \\
T \;:\;& \text{maximum route duration (total travel time and service time)}. \\
f_i \;:\;& \text{service frequency of customer } i. \\
a_{pl} \;:\;& \text{binary constant; } 1 \text{ iff day } l \text{ belongs to visit combination } p,\ 0 \text{ otherwise}. \\
\tau_o = 0,\ \forall v_o \in V^{\mathrm{DEP}} \;:\;& \text{convention: zero service time at depots}.
\end{align*}$$

## Decision Variables

$$\begin{align*}
y_{ipo} \in \{0,1\} \;:\;& 1 \text{ iff customer } i \text{ is assigned to visit combination } p \in L_i \\
                        & \text{and depot } o \in V^{\mathrm{DEP}};\ 0 \text{ otherwise}. \\
x_{ijklo} \in \{0,1\} \;:\;& 1 \text{ iff vehicle } k \text{ coming from depot } o \text{ on day } l \\
                          & \text{visits } v_j \text{ immediately after } v_i;\ 0 \text{ otherwise}.
\end{align*}$$

## Objective

$$\begin{equation}
\text{Minimize} \quad \sum_{v_i \in V} \sum_{v_j \in V} \sum_{k=1}^{m} \sum_{l=1}^{t} \sum_{v_o \in V^{\mathrm{DEP}}} c_{ij}\, x_{ijklo}
\tag{5}
\end{equation}$$

## Constraints

$$\begin{align}
& \sum_{p \in L_i} \sum_{v_o \in V^{\mathrm{DEP}}} y_{ipo} = 1
  && v_i \in V^{\mathrm{CST}} \tag{6} \\[4pt]
& \sum_{v_j \in V} \sum_{k=1}^{m} x_{ijklo} - \sum_{p \in L_i} a_{pl}\, y_{ipo} = 0
  && v_i \in V^{\mathrm{CST}};\ v_o \in V^{\mathrm{DEP}};\ l = 1 \ldots t \tag{7} \\[4pt]
& \sum_{v_j \in V} x_{ojklo} \leq 1
  && v_o \in V^{\mathrm{DEP}};\ k = 1 \ldots m;\ l = 1 \ldots t \tag{8} \\[4pt]
& \sum_{v_j \in V} x_{ijklo} = 0
  && v_i \in V^{\mathrm{DEP}};\ v_o \in V^{\mathrm{DEP}};\ v_o \neq v_i; \nonumber \\
& && \quad k = 1 \ldots m;\ l = 1 \ldots t \tag{9} \\[4pt]
& \sum_{v_j \in V} x_{jiklo} - \sum_{v_j \in V} x_{ijklo} = 0
  && v_i \in V;\ v_o \in V^{\mathrm{DEP}};\ k = 1 \ldots m;\ l = 1 \ldots t \tag{10} \\[4pt]
& \sum_{v_i \in V} \sum_{v_j \in V} q_i\, x_{ijklo} \leq Q
  && v_o \in V^{\mathrm{DEP}};\ k = 1 \ldots m;\ l = 1 \ldots t \tag{11} \\[4pt]
& \sum_{v_i \in V} \sum_{v_j \in V} (c_{ij} + \tau_i)\, x_{ijklo} \leq T
  && v_o \in V^{\mathrm{DEP}};\ k = 1 \ldots m;\ l = 1 \ldots t \tag{12} \\[4pt]
& \sum_{v_i \in S} \sum_{v_j \in S} x_{ijklo} \leq |S| - 1
  && S \subseteq V^{\mathrm{CST}};\ |S| \geq 2;\ v_o \in V^{\mathrm{DEP}}; \nonumber \\
& && \quad k = 1 \ldots m;\ l = 1 \ldots t \tag{13} \\[4pt]
& x_{ijklo} \in \{0, 1\}
  && v_i \in V;\ v_j \in V;\ v_o \in V^{\mathrm{DEP}}; \nonumber \\
& && \quad k = 1 \ldots m;\ l = 1 \ldots t \tag{14} \\[4pt]
& y_{ipo} \in \{0, 1\}
  && v_i \in V;\ p \in L_i;\ v_o \in V^{\mathrm{DEP}} \tag{15}
\end{align}$$
