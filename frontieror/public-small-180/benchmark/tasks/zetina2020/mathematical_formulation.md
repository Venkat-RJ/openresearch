# Original Formulation: Maximum Cut (MaxCut) and Travelling Salesman Problem (TSP)

*Source: Efficient Internal Strategies in Quantum Relaxation based Branch-and-Bound, Hiromichi Matsuyama, Wei-hao Huang, Kohji Nishimura, Yu Yamashiro (Jij Inc.), 2024 (arXiv:2405.00935).*

The paper studies two classical combinatorial optimization problems within its Quantum Relaxation based Branch-and-Bound (QR-BnB) framework: an unconstrained problem (MaxCut) and a constrained problem (TSP). It first states the generic binary quadratic program (5) that the branch-and-bound operates on, then gives the concrete MaxCut (7) and TSP (8) formulations whose notation is carried into the experiments. Both problems are part of the benchmark and are presented below.

## Sets and Indices

- Generic problem: $I=\{1,\dots,n\}$ index set of the $n$ binary variables; $m$ constraints.

- MaxCut: undirected graph $G(V,E)$ with node set $V$ and edge set $E$; $(i,j)\in E$.

- TSP: $N$ cities; city index $i\in\{0,1,\dots,N-1\}$; time-step index $t\in\{0,1,\dots,N-1\}$.

## Parameters

- $Q\in\mathbb{R}^{n\times n}$: objective coefficient matrix of the generic binary quadratic problem.

- $A\in\mathbb{R}^{m\times n}$, $b\in\mathbb{R}^{m}$: constraint matrix and right-hand side of the generic problem.

- $d_{ij}$: distance between city $i$ and city $j$ (TSP), Euclidean, symmetric.

- $A,\,B$: penalty weights in the TSP QUBO reformulation (set $A=B=1$ in the experiments).

## Decision Variables

- $\mathbf{x}\in\{0,1\}^{n}$: binary decision vector of the generic problem.

- $s_i\in\{-1,1\}$ for all $i\in V$: spin variable indicating the subset assignment of node $i$ (MaxCut).

- $x_{i,t}\in\{0,1\}$: equals $1$ if the salesman is in city $i$ at time step $t$, $0$ otherwise (TSP).

## Objective and Constraints

#### Generic binary quadratic problem.

The branch-and-bound is defined over the generic problem $$\begin{align}
  S:\quad z^{*} = \min\bigl\{\, \mathbf{x}^{\mathsf{T}} Q\, \mathbf{x} \;:\;
       \mathbf{x}\in X \,\bigr\}, \tag{5}
\end{align}$$ where $X=\{\mathbf{x}\in\{0,1\}^{n} : A\mathbf{x}\le b\}$ is the feasible set, with $b\in\mathbb{R}^{m}$, $A\in\mathbb{R}^{m\times n}$, $Q\in\mathbb{R}^{n\times n}$.

#### MaxCut problem (unconstrained).

$$\begin{align}
  \max_{s}\;\; \frac{1}{2}\sum_{(i,j)\in E}\bigl(1 - s_i s_j\bigr), \tag{7}
\end{align}$$ with the spin variable $s_i\in\{-1,1\}$ indicating the assignment to each subset. There are no constraints (every assignment of nodes to the two subsets is feasible).

#### TSP problem (constrained).

$$\begin{align}
  \min_{x}\;\; & \sum_{t}\sum_{ij} d_{ij}\, x_{i,t}\, x_{j,\,t+1 \bmod N} \notag\\
  \text{s.t.}\;\; & \sum_{i=0}^{N-1} x_{i,t} = 1 && \forall t\in\{0,1,\dots,N-1\}, \tag{8}\\
                  & \sum_{t=0}^{N-1} x_{i,t} = 1 && \forall i\in\{0,1,\dots,N-1\}, \notag
\end{align}$$ using the variable $x_{i,t}\in\{0,1\}$, which is $1$ when the salesman is in city $i$ at time $t$. The objective uses modular arithmetic $t+1 \bmod N$ to close the tour. The two equality constraints are the city-per-time and time-per-city one-hot constraints.
