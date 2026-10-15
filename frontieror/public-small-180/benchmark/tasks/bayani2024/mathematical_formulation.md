# Original Formulation: Quadratic Semi-Assignment Problem (QSAP)

*Source: A dual bounding framework for binary quadratic combinatorial optimization, M. Bayani, B. Rostami, Y. Adulyasak, L.-M. Rousseau, 2021 (Les Cahiers du GERAD, G–2021–43).*

## Sets and Indices

- $N = \{1, \dots, n\}$: set of clients.

- $H = \{1, \dots, h\}$: set of servers.

- $A$: set of edges (client–server assignment pairs). An edge $e = (i,j)$, $i \in N$, $j \in H$, represents assigning client $i$ to server $j$. ($A$ is the set of edges with one endpoint in $H$; here, with $N$ the clients and $H$ the servers, $A = E$.)

- $\delta(i) \subseteq A$: set of edges incident to client $i \in N$ (i.e. all assignments of client $i$).

- $\mathcal{E} = E \times E$: set of all (ordered) pairs of edges.

## Parameters

- $c_e = c_{ij} \in \mathbb{R}$: linear cost of assigning client $i$ to server $j$ (edge $e = (i,j)$).

- $q_{ef} = q_{ijkl} \in \mathbb{R}$: quadratic cost incurred when client $i$ is assigned to server $j$ *and* client $k$ is assigned to server $l$ simultaneously, for the pair of edges $e = (i,j)$, $f = (k,l)$.

## Decision Variables

- $x_e \in \{0,1\}$: equals $1$ if the assignment corresponding to edge $e = (i,j)$ is selected (client $i$ assigned to server $j$), $0$ otherwise.

## Objective

$$\begin{align}
\min \quad & \sum_{e \in A} c_e x_e + \sum_{(e,f) \in \mathcal{E}} q_{ef}\, x_e x_f \tag{12}
\end{align}$$

## Constraints

$$\begin{align}
& \sum_{e \in \delta(i)} x_e = 1 && \forall i \in N \tag{13} \\
& x_e \in \{0,1\} && \forall e \in A \tag{14}
\end{align}$$
