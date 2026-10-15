# Original Formulation: Two-Machine Small-Bucket Lot-Sizing with Backlogging and Start-Ups (Con-NT)

*Source: bc–prod: A Specialized Branch-and-Cut System for Lot-Sizing Problems, Gaetan Belvaux and Laurence A. Wolsey, 2000 (Management Science 46(5):724–738).*

The paper describes the prototype system *bc–prod* via a generic lot-sizing template (Constraints (1)–(5)) and two concrete problem instances. The instance studied here is **Example 2 (Con-NT)**, the two-machine small-bucket model with backlogging and start-up costs/times stated in Eqs. (9)–(14); its variable symbols and reported objective are the ones carried into the computational experiments. The generic template and the big-bucket instance (set1ch) are recorded under **Variants**.

## Sets and Indices

- $i = 1, \dots, NI$ : items (products).

- $k = 1, \dots, NK$ : machines (Con-NT uses two machines).

- $t = 1, \dots, NT$ : time periods ($NT = 12/24$ in the instance).

## Parameters

- $f$ : set-up cost (uniform over all items and machines).

- $g$ : start-up cost (uniform over all items and machines).

- $h^i$ : per-unit inventory holding cost for item $i$.

- $e^i$ : per-unit backlogging penalty cost for item $i$.

- $d^i_t$ : prespecified demand for item $i$ in period $t$.

- $C^k$ : production capacity of machine $k$ (constant over $t$).

- $\sigma^k$ : start-up time – loss of production capacity on machine $k$ when a start-up occurs.

## Decision Variables

- $x^{ik}_t \geq 0$ : production of item $i$ on machine $k$ in period $t$ (continuous).

- $s^i_t \geq 0$ : stock of item $i$ at the end of period $t$ (continuous).

- $r^i_t \geq 0$ : backlog of item $i$ at the end of period $t$ (continuous).

- $y^{ik}_t \in \{0,1\}$ : set-up variable; $1$ if machine $k$ is set up to produce item $i$ in period $t$.

- $z^{ik}_t \in \{0,1\}$ : start-up variable; $1$ if machine $k$ starts a sequence of set-ups of item $i$ in period $t$.

## Objective

$$\begin{align}
\min \quad \sum_{i,k,t} \bigl( f\,y^{ik}_t + g\,z^{ik}_t \bigr)
        + \sum_{i,t} \bigl( h^i\,s^i_t + e^i\,r^i_t \bigr) \notag
\end{align}$$

## Constraints

$$\begin{align}
s^i_{t-1} - r^i_{t-1} + \sum_{k} x^{ik}_t
   &= d^i_t + s^i_t - r^i_t && \forall\, i,\,t \tag{9}\\[2pt]
x^{ik}_t + \sigma^k z^{ik}_t
   &\leq C^k y^{ik}_t && \forall\, i,\,k,\,t \tag{10}\\[2pt]
z^{ik}_t &\geq y^{ik}_t - y^{ik}_{t-1} && \forall\, i,\,k,\,t \tag{11}\\[2pt]
\sum_{i} y^{ik}_t &\leq 1 && \forall\, k,\,t \tag{12}\\[2pt]
z^{ik}_t &\leq y^{ik}_t && \forall\, i,\,k,\,t \tag{13}\\[2pt]
y^{ik}_{t-1} + z^{ik}_t + \sum_{j:\, j \neq i} \bigl( y^{jk}_t - z^{jk}_t \bigr)
   &\leq 1 && \forall\, i,\,k,\,t \tag{14}
\end{align}$$ with domains and initial conditions $$\begin{align}
s,\, r,\, x &\geq 0, \qquad y,\, z \in \{0,1\}, \notag\\
s^i_0 &= r^i_0 = 0. \notag
\end{align}$$
