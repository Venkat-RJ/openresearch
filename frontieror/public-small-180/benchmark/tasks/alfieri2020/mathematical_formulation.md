# Original Formulation: Graph Coloring Problem (GCP-compact)

*Source: Enhancing Column Generation by a Machine-Learning-Based Pricing Heuristic for Graph Coloring, Yunzhuang Shen, Yuan Sun, Xiaodong Li, Andrew Eberhard, Andreas Ernst, 2022 (AAAI).*

## Sets and Indices

- $G=(\mathcal{V},\mathcal{E})$: an undirected graph.

- $\mathcal{V}$: set of vertices, indexed by $i$ (and $j$).

- $\mathcal{E}$: set of edges; an edge is an unordered pair $(i,j)$.

- $\mathcal{C}$: set of candidate colors, indexed by $c$.

## Parameters

The instance is fully specified by the graph $G=(\mathcal{V},\mathcal{E})$. The color palette $\mathcal{C}$ is a candidate set of colors; its cardinality $|\mathcal{C}|$ is an upper bound on the chromatic number (in practice obtained by a greedy coloring upper bound).

## Decision Variables

- $x_{i,c}\in\{0,1\}$: $1$ if vertex $i$ is assigned color $c$, $0$ otherwise.

- $z_c\in\{0,1\}$: $1$ if color $c$ is used (in use), $0$ otherwise.

## Objective

$$\begin{equation}
\min_{\mathbf{x},\mathbf{z}}\ \sum_{c\in\mathcal{C}} z_c
\tag{GCP-compact, 1}
\end{equation}$$

## Constraints

$$\begin{align}
\sum_{c\in\mathcal{C}} x_{i,c} &= 1, & & i\in\mathcal{V}, \tag{2}\\[2pt]
x_{i,c} + x_{j,c} &\le z_c, & & (i,j)\in\mathcal{E};\ c\in\mathcal{C}, \tag{3}\\[2pt]
x_{i,c} &\in \{0,1\}, & & i\in\mathcal{V};\ c\in\mathcal{C}, \tag{4}\\[2pt]
z_c &\in \{0,1\}, & & c\in\mathcal{C}. \tag{5}
\end{align}$$
