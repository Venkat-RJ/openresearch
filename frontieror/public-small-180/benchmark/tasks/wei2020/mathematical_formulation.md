# Original Formulation: One-Dimensional Bin-Packing Problem (1D-BPP / 1D-BPPC)

*Source: “A New Branch-and-Price-and-Cut Algorithm for One-Dimensional Bin-Packing Problems,” Lijun Wei, Zhixing Luo, Roberto Baldacci, Andrew Lim, INFORMS Journal on Computing, 2019.*

The paper introduces, in Section 2 (“Set-Partitioning Formulation with SR Inequalities”), the set-partitioning model with subset-row (SR) inequalities, called formulation $F$. This is the first and only formulation the paper writes as the definition of the problem it studies; its variables ($x_p$) and notation are carried into the pricing problems, the branch-and-price-and-cut algorithm, and the computational results. It applies to both the 1D-BPP (empty conflict set) and the 1D-BPPC (nonempty conflict graph), which differ only in the definition of the feasible pattern set $\mathscr{P}$.

## Sets and Indices

- $N = \{1, 2, \ldots, n\}$: set of $n$ items.

- $G = (N, E)$: undirected conflict graph (1D-BPPC). $E$ is a set of edges such that $\{i,j\} \in E$ when items $i$ and $j$ are in conflict (cannot share a bin). For the plain 1D-BPP, $E = \emptyset$.

- A *pattern* $S$ is a subset of $N$ such that $\sum_{j \in S} w_j \le c$. For the 1D-BPPC, a pattern $S$ cannot contain any pair $i,j$, $i \ne j$, with $\{i,j\} \in E$.

- $\mathscr{P}$: index set of all feasible patterns for the 1D-BPP (1D-BPPC).

- $N_p$: set of items of pattern $p \in \mathscr{P}$.

- $\mathscr{C} \subseteq \{S \subset N : |S| = 3\}$: a subset of all item triplets (cardinality-3 item sets).

- $\mathscr{P}(S) \subseteq \mathscr{P}$: subset of patterns containing at least two items of $S$, i.e., $\mathscr{P}(S) = \{p \in \mathscr{P} : |N_p \cap S| \ge 2\}$.

## Parameters

- $w_j$: integer weight of item $j$, for $j \in N$.

- $c$: integer capacity of each (identical) bin.

- $a_{ip} \in \{0,1\}$: $(0,1)$ coefficient equal to $1$ if item $i \in N_p$, and $0$ otherwise, for $i \in N$, $p \in \mathscr{P}$.

## Decision Variables

- $x_p \in \{0,1\}$, for $p \in \mathscr{P}$: binary variable equal to $1$ if and only if pattern $p$ is in the optimal solution.

## Objective

$$\begin{align}
  z(F) \;=\; \min \; \sum_{p \in \mathscr{P}} x_p, \tag{1}
\end{align}$$ which minimizes the number of bins used.

## Constraints

$$\begin{align}
  \sum_{p \in \mathscr{P}} a_{ip}\, x_p &= 1,
      && \forall\, i \in N, \tag{2}\\[4pt]
  \sum_{p \in \mathscr{P}(S)} x_p &\le 1,
      && \forall\, S \in \mathscr{C}, \tag{3}\\[4pt]
  x_p &\in \{0,1\},
      && \forall\, p \in \mathscr{P}. \notag
\end{align}$$

Constraints (2) ensure that each item $i \in N$ is assigned to exactly one bin. Constraints (3) are the subset-row (SR) inequalities, applied to subsets of items of cardinality three; they are Chvátal–Gomory rank-1 cuts that are valid for the set-partitioning model. (Formulation $F'$ is $F$ without the SR inequalities (3).)
