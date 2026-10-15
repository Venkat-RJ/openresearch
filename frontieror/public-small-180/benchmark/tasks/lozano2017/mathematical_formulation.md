# Original Formulation: Bilevel Mixed-Integer Programming Problem (BMIP / BMILP)

*Source*: Lozano & Smith (2017), “A Value-Function-Based Exact Approach for the Bilevel Mixed-Integer Programming Problem,” *Operations Research*. Equations (1a)–(1d) give the general BMIP; Equations (14a)–(14d) give the linear special case (BMILP). Both are transcribed below in bilevel form without KKT / value-function reformulation.

## Sets, Indices, and Domains

- $n_1$: number of leader variables; $n_2$: number of follower variables.

- $m_1$: number of upper-level constraints; $m_2$: number of lower-level constraints.

- $\mathcal{I} \subseteq \{1,\ldots,n_1\}$: indices of integer leader variables.

- $\mathcal{J} \subseteq \{1,\ldots,n_2\}$: indices of integer follower variables.

- $\mathcal{H}^x = \{\mathbf{x} \ge \mathbf{0} :\ x_i \in \mathbb{Z},
          \ \forall\, i \in \mathcal{I}\}$.

- $\mathcal{H}^y = \{\mathbf{y} \ge \mathbf{0} :\ y_j \in \mathbb{Z},
          \ \forall\, j \in \mathcal{J}\}$.

- Functions $\phi^l,\phi^f,g_j^k,h_j^k$ are continuous over $\mathcal{H}^x \times \mathcal{H}^y$ ($k=1,2$; $j=1,\ldots,m_k$).

## Decision Variables

- $\mathbf{x} \in \mathcal{H}^x$: leader (upper-level) decision vector.

- $\mathbf{y} \in \mathcal{H}^y$: follower (lower-level) decision vector.

## General BMIP (Equations 1a–1d)

$$\begin{align}
  z^{*} \;=\; \max_{\mathbf{x},\mathbf{y}}\; & \phi^l(\mathbf{x},\mathbf{y})
    \tag{1a}\\[2pt]
  \text{s.t.}\quad
    & g_j^1(\mathbf{x}) + h_j^1(\mathbf{y}) \;\le\; b_j^1,
      \quad \forall\, j = 1,\ldots,m_1,
    \tag{1b}\\[2pt]
    & \mathbf{y} \;\in\;
       \arg\max_{\mathbf{y}^f}\Bigl\{\; \phi^f(\mathbf{x},\mathbf{y}^f) \;\Big|\;
         g_j^2(\mathbf{x}) + h_j^2(\mathbf{y}^f) \le b_j^2,\
         \forall\, j=1,\ldots,m_2;\;
         \mathbf{y}^f \in \mathcal{H}^y \,\Bigr\},
    \tag{1c}\\[2pt]
    & \mathbf{x} \;\in\; \mathcal{H}^x.
    \tag{1d}
\end{align}$$ Equation (1c) encodes the optimistic assumption (the follower breaks ties in favor of the leader).

## Linear Special Case: BMILP (Equations 14a–14d)

Given conforming coefficient matrices and vectors $\mathbf{c},\mathbf{d}^1,
\mathbf{d}^2,\mathbf{A}^1,\mathbf{A}^2,\mathbf{B}^1,\mathbf{B}^2,
\mathbf{b}^1,\mathbf{b}^2$: $$\begin{align}
  \max_{\mathbf{x},\mathbf{y}}\;
    & \mathbf{c}\,\mathbf{x} + \mathbf{d}^1 \mathbf{y}
    \tag{14a}\\[2pt]
  \text{s.t.}\quad
    & \mathbf{A}^1 \mathbf{x} + \mathbf{B}^1 \mathbf{y} \;\le\; \mathbf{b}^1,
    \tag{14b}\\[2pt]
    & \mathbf{y} \;\in\;
       \arg\max_{\mathbf{y}^f}\Bigl\{\; \mathbf{d}^2 \mathbf{y}^f \;\Big|\;
         \mathbf{A}^2 \mathbf{x} + \mathbf{B}^2 \mathbf{y}^f \;\le\; \mathbf{b}^2;\;
         \mathbf{y}^f \in \mathcal{H}^y \,\Bigr\},
    \tag{14c}\\[2pt]
    & \mathbf{x} \;\in\; \mathcal{H}^x.
    \tag{14d}
\end{align}$$

## Released benchmark specialization

For every bundled instance, $\mathbf{d}^1=\mathbf{0}$, so (14a) reduces to
$\max\;\mathbf{c}\mathbf{x}$. The instance flag
`leader_follower_objective_coefficients_fixed_zero=true` is part of the
contract. Entries of `integer_follower_vars` are one-based.

## Assumptions (reproduction-critical)

1.  Both upper- and lower-level feasible regions are compact sets.

2.  $g_j^2(\mathbf{x})$ is integer-valued for all $\mathbf{x} \in \mathcal{H}^x$, $j = 1,\ldots,m_2$.

3.  All leader variables are integer-valued.
