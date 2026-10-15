# Original Formulation (General Min-Max-Min, Eq. 1)

## Sets, Dimensions, and Parameters

- Dimensions: $N = p + p'$ first-stage vars; $N_1$ binary first-stage; $M = q + q'$ second-stage vars; $M_1$ binary second-stage; $S$ dimension of uncertainty vector $\xi$; $S'$ rows of $A$.

- First-stage data: $G \in \mathbb{Z}^{N' \times N}$, $g \in \mathbb{Z}^{N'}$, $c \in \mathbb{Z}^N$.

- Second-stage data: $E \in \mathbb{Z}^{M'' \times M}$, $e \in \mathbb{Z}^{M''}$, $f \in \mathbb{Z}^M$, $Q \in \mathbb{Z}^{M \times S}$.

- Linking-constraint data: $H \in \mathbb{Z}^{M' \times M}$, $T \in \mathbb{Z}^{M' \times N_1}$, $d \in \mathbb{Z}^{M'}$.

- Uncertainty-set data: $A \in \mathbb{Z}^{S' \times S}$, $b \in \mathbb{Z}^{S'}$.

## Feasible Regions

$$\begin{align*}
X     & = \{\, x \in \{0,1\}^{N_1} \times \mathbb{R}^{N - N_1}_+ \;:\; G x \leq g \,\}, \\
Y     & = \{\, y \in \{0,1\}^{M_1} \times \mathbb{R}^{M - M_1}_+ \;:\; E y \leq e \,\}, \\
Y(x)  & = \{\, y \in Y \;:\; H y \leq d - T x_1 \,\}, \\
\Xi   & = \{\, \xi \in \mathbb{R}^{S} \;:\; A \xi \leq b \,\}.
\end{align*}$$

## Original Problem

$$\begin{align}
\min_{x \in X} \; c^\top x \;+\; \max_{\xi \in \Xi} \; \min_{y \in Y(x)} \; (f + Q\xi)^\top y
  \tag{1}
\end{align}$$

At the first stage, $x$ is chosen before $\xi$ is observed. An adversary then picks $\xi \in \Xi$ to maximize the second-stage cost. Finally, the decision maker reacts by choosing $y \in Y(x)$ that minimizes $(f + Q\xi)^\top y$. Both $X$ and $Y$ are bounded mixed-binary sets.

# Benchmark Application: LP-Relaxed Two-Stage Robust Knapsack

The original paper's Eq. (43) has binary recourse. The released benchmark
uses the explicit LP relaxation below, replacing the binary domains of
$(y,r)$ by their unit intervals. This distinction is material and defines the
objective values in the bundled references.

## Sets and Parameters

- $I = \{1, \ldots, I\}$: set of items.

- $c_i$: weight (capacity requirement) of item $i$.

- $\bar{p}_i$: expected profit of item $i$; $\hat{p}_i$: maximum profit degradation.

- $f_i$: outsourcing cost/penalty; $t_i$: additional repair capacity for item $i$.

- $C$: total knapsack capacity; $\Gamma$: uncertainty budget.

## Uncertainty Set

$$\begin{align*}
\Xi \;=\; \Big\{\, \xi \in \mathbb{R}^{I}_+ \;:\; \sum_{i \in I} \xi_i \leq \Gamma,\; 0 \leq \xi_i \leq 1 \ \forall i \in I \,\Big\}.
\end{align*}$$

## Decision Variables

First-stage: $x_i \in \{0,1\}$, $x_i = 1$ iff item $i$ is selected for production.  
Second-stage (in $Y_{LP}(x)$): $0\le y_i\le1$ (fraction produced as-is), $0\le r_i\le1$ (fraction repaired). $$\begin{align*}
Y_{LP}(x) \;=\; \Big\{ (y, r) \in [0,1]^I \times [0,1]^I \;:\;
  & \sum_{i \in I} (c_i y_i + t_i r_i) \leq C,\;
    y_i \leq x_i\ \forall i,\;
    r_i \leq y_i\ \forall i \Big\}.
\end{align*}$$

## LP-Relaxed Min-Max-Min Problem

$$\begin{align}
\min_{x \in \{0,1\}^I}\; \sum_{i \in I} (f_i - \bar{p}_i)\, x_i
  \;+\; \max_{\xi \in \Xi}\; \min_{(y,r) \in Y_{LP}(x)}\;
    \sum_{i \in I} \Big[\, (\hat{p}_i\, \xi_i - f_i)\, y_i \;-\; \hat{p}_i\, \xi_i\, r_i \,\Big]
  \tag{43-LP}
\end{align}$$
