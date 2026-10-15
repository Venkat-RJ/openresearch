# Original Formulation: Multilocation Transshipment Problem (2-SLP with Random Cost Coefficients)

*Source: Stochastic Decomposition for Two-Stage Stochastic Linear Programs with Random Cost Coefficients, Harsha Gangammanavar, Yifan Liu, and Suvrajeet Sen, INFORMS Journal on Computing, 2020.*

The paper studies the generic two-stage stochastic linear program (2-SLP) with random cost coefficients, stated first in Section 1 as Eqs. (1a)–(1b). The concrete instance that the benchmark implements is the Multilocation Transshipment Problem of Appendix C.3 (Eqs. (C.1)–(C.2)), a direct specialization of (1a)–(1b). The canonical *Objective*/*Constraints* block below is the transshipment instance; the generic 2-SLP that defines its structure is reproduced under *Remarks*.

## Sets and Indices

- $N$ : number of nonidentical retailers (suppliers); locations indexed by $i,j \in \{1,\dots,N\}$.

- $\tilde{\omega}$ : random variable defined on probability space $(\Omega,\mathcal{F},\mathbb{P})$, with realization $\omega$; the joint distribution of demand and transshipment cost is given as a finite discrete scenario set.

## Parameters

- $h_i$ : unit inventory holding cost at retailer $i$.

- $p_i$ : unit backlog (penalty) cost at retailer $i$.

- $c_{ij}(\omega)$ : realized unit transshipment cost from retailer $i$ to retailer $j$ ($i \neq j$); deterministic (equal to its mean) for most pairs and $\mathcal{N}(c_{ij}^{\text{mean}}, \sigma_{ij})$ with $\sigma_{ij} = 0.2\, c_{ij}^{\text{mean}}$ for a designated subset of random-cost links.

- $d_i(\tilde{\omega})$ : realized (random) demand at retailer $i$, with the demand vector following a finite discrete distribution. Demand and cost randomness are independent.

## Decision Variables

**First stage** (here-and-now, before $\omega$ is observed):

- $s_i \geq 0$ : order-up-to stock (replenishment) level at retailer $i$.

**Second stage** (recourse, per realization $\omega$):

- $f_i \geq 0$ : on-hand stock at $i$ used to satisfy local demand at $i$.

- $e_i \geq 0$ : ending inventory at $i$ after local use and outbound shipments.

- $r_i \geq 0$ : backlog (unmet demand) at $i$.

- $q_i \geq 0$ : replenishment quantity increasing inventory back to $s_i$ after replenishment.

- $t_{ij} \geq 0$ : units transshipped from retailer $i$ to retailer $j$ ($i \neq j$).

## Objective

$$\begin{align}
\min_{s_i \geq 0,\; i=1,\ldots,N} \quad & \mathbb{E}\big[\, h\big((s_i)_i, \tilde{\omega}\big) \,\big] \tag{C.1}
\end{align}$$ where the recourse cost is $$\begin{align}
h\big((s_i)_i, \tilde{\omega}\big) := \min \quad &
\sum_{i=1}^{N} \big( h_i e_i + p_i r_i \big)
+ \sum_{\substack{i,j=1 \\ i \neq j}}^{N} c_{ij}(\omega)\, t_{ij} \tag{C.2}
\end{align}$$

## Constraints

The recourse problem (C.2) is, for a given $(s_i)_i$ and realization $\omega$: $$\begin{align}
& f_i + \sum_{\substack{j=1 \\ j \neq i}}^{N} t_{ij} + e_i = s_i,
    && i = 1,\ldots,N; \tag{C.2a}\\
& f_i + \sum_{\substack{j=1 \\ j \neq i}}^{N} t_{ji} + r_i = d_i(\tilde{\omega}),
    && i = 1,\ldots,N; \tag{C.2b}\\
& \sum_{i=1}^{N} \big( r_i + q_i \big) = \sum_{i=1}^{N} d_i(\tilde{\omega}); \tag{C.2c}\\
& e_i + q_i = s_i, && i = 1,\ldots,N; \tag{C.2d}\\
& e_i,\, r_i,\, q_i,\, t_{ij} \geq 0, && i,j = 1,\ldots,N. \tag{C.2e}
\end{align}$$
