# Released Formulation: Integrated-Primal UCGNA MISOCP

*Source: Benders Subproblem Decomposition for Bilevel Problems with Convex Follower, Byeon and Van Hentenryck, 2022.*

The source paper studies a bilevel second-order-cone model.  The executable
FrontierOR release is deliberately narrower and every instance declares
`released_problem_variant = integrated_primal_ucgna`.  It jointly optimizes
all commitment, dispatch, gas-network, gas-price, and electricity-network
variables under the leader constraints (40b--41e) and follower **primal**
constraints (42b--42w):

$$
\min\; \beta C_{\mathrm{electricity}}
       +(1-\beta)C_{\mathrm{gas}},
$$

where $\beta$ is the instance field `objective_weight_beta`.  The electricity
term contains no-load, startup, and bid-dispatch costs; the gas term contains
gas-demand shedding and interval supply costs.  The conic Weymouth and heat
rate inequalities are retained.  Zonal gas prices are bounded primal decisions
linked to commitment by the released bid-validity constraints.

There is no follower `argmin`, dual-feasibility system, or strong-duality
constraint in this executable variant.  In particular, the former incomplete
SOC dual is not part of the released feasible set.  A valid solution reports
the complete primary-variable key set, including explicit zeros, and the
checker recomputes this objective and every released primal constraint.

## Source-paper bilevel context (not the executable release contract)

## Sets and Parameters

- $n_x, n_y$: dimensions of the leader and follower variable vectors.

- $m_x, m_y$: numbers of leader and follower constraints.

- $\mathcal{I} \subseteq \{1,\dots,n_x\}$: index set of integer leader variables.

- $\mathcal{J} \subseteq \mathcal{I}$: indices $i$ such that column $i$ of $A$ is nonzero (leader variables that appear in the follower problem).

- $c_x \in \mathbb{R}^{n_x}$, $c_y \in \mathbb{R}^{n_y}$: leader objective coefficients.

- $G_x \in \mathbb{R}^{m_x \times n_x}$, $G_y \in \mathbb{R}^{m_x \times n_y}$, $h \in \mathbb{R}^{m_x}$: leader constraint data.

- $A \in \mathbb{R}^{m_y \times n_x}$, $B \in \mathbb{R}^{m_y \times n_y}$, $b \in \mathbb{R}^{m_y}$, $d \in \mathbb{R}^{n_y}$: follower constraint and objective data.

- $\underline{x}_i, \overline{x}_i$: lower and upper bounds on $x_i$ (finite for $i \in \mathcal{J}$).

- $\mathcal{K}_x, \mathcal{K}_y$: Cartesian products of second-order cones and nonnegative orthants (the ambient cones for $x$ and $y$).

## Decision Variables

- $x \in \mathbb{R}^{n_x}$: leader (upper-level) decision variables.

- $y \in \mathbb{R}^{n_y}$: follower (lower-level) decision variables.

## Objective

$$\begin{equation}
\min_{x,\,y} \quad c_x^{\top} x + c_y^{\top} y \tag{1a}
\end{equation}$$

## Constraints (Bilevel)

$$\begin{align}
G_x x + G_y y & \;\geq\; h, \tag{1b} \\[2pt]
x & \;\in\; \mathcal{X} \;:=\; \Bigl\{ x \in \mathcal{K}_x \;:\;
      x_i \in [\underline{x}_i,\overline{x}_i] \cap \mathbb{Z},\;\forall i \in \mathcal{I} \Bigr\}, \tag{1c} \\[2pt]
y & \;\in\; \arg\min_{y' \in \mathcal{K}_y}
      \Bigl\{\, d^{\top} y' \;:\; A x + B y' \geq b \,\Bigr\}. \tag{1d}
\end{align}$$

Constraint (1d) enforces that $y$ is an optimal response of the follower to the leader decision $x$ (optimistic bilevel); the problem is an MISOCP-follower bilevel program. Under Assumption 2(b) integer bounded $x_i$ for $i \in \mathcal{J}$ may be encoded as binary without loss of generality.
