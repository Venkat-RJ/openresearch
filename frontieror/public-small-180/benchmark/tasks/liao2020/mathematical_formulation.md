# Original Formulation: Distributionally Robust Outpatient Colonoscopy Scheduling (DROCS)

*Source*: Shehadeh, Cohn, Jiang (2020), “A distributionally robust optimization approach for outpatient colonoscopy scheduling,” *EJOR*. First-stage variables satisfy assignment (1) and scheduling (2); the outer problem is the distributionally robust min–max given by Eq. (6). The Prop. 4 MILP reformulation is *not* substituted here.

**Benchmark cost selector.** If the JSON instance provides multiple named cost
structures, the first declared entry (`Cost1` in the released data) supplies
$c^w$, $c^g$, and $c^o$ for this task.

## Sets and Indices

- $p \in \{1,\ldots,P\}$: patients / procedures.

- $i \in \{1,\ldots,P\}$: positions in the appointment sequence.

## Parameters

- $c^w_i$: unit waiting-time cost of appointment $i$.

- $c^g_i$: unit provider idle-time cost between appointments $i-1$ and $i$.

- $c^o$: unit provider overtime cost.

- $L$: scheduled service-hour length of the provider.

- $\mu^A_p,\,\mu^I_p$: mean procedure durations with adequate / inadequate bowel prep for patient $p$.

- $\mu^u_p$: mean arrival-time deviation of patient $p$.

- $\mu^q_p$: probability of adequate prep for patient $p$.

- $d^{AL}_p,d^{AU}_p$: lower / upper bounds of procedure duration with adequate prep.

- $d^{IL}_p,d^{IU}_p$: lower / upper bounds of procedure duration with inadequate prep.

- $u^L_p,u^U_p$: lower / upper bounds of the arrival-time deviation.

## Decision Variables (first stage)

- $x_{p,i} \in \{0,1\}$: $1$ if patient $p$ is assigned to appointment position $i$.

- $t_i \ge 0$: scheduled start time of appointment $i$, with $t_1 = 0$.

## Random Parameters and Ambiguity Set

Let $\xi = (d^A, d^I, u, q)$ denote the random procedure durations (adequate and inadequate prep), arrival deviations, and bowel-prep adequacy indicators, with support $S = S^A \times S^I \times S^u \times S^q$ where $$\begin{align*}
  S^A &= \{d^A \ge 0 : d^{AL}_p \le d^A_p \le d^{AU}_p,\ d^A_{P+1}=0\},\\
  S^I &= \{d^I \ge 0 : d^{IL}_p \le d^I_p \le d^{IU}_p,\ d^I_{P+1}=0\},\\
  S^u &= \{u : u^L_p \le u_p \le u^U_p,\ u_{P+1}=0\},\qquad
  S^q  = \{0,1\}^{P}.
\end{align*}$$ The ambiguity set is $$\begin{equation}
  \mathcal{F}(S,\mu) \;=\;
    \Bigl\{\, P \in \mathcal{P}(S) \;\Big|\;
      \mathbb{E}_P[d^A_p] = \mu^A_p,\;
      \mathbb{E}_P[d^I_p] = \mu^I_p,\;
      \mathbb{E}_P[u_p]   = \mu^u_p,\;
      \mathbb{E}_P[q_p]   = \mu^q_p,\ \forall p \,\Bigr\}.
  \tag{A}
\end{equation}$$

## First-Stage Constraints

#### Assignment constraints (1).

$$\begin{align}
  \sum_{p=1}^{P} x_{p,i} &\;=\; 1,
    && \forall\, i = 1,\ldots,P, \tag{1a}\\[2pt]
  \sum_{i=1}^{P} x_{p,i} &\;=\; 1,
    && \forall\, p = 1,\ldots,P, \tag{1b}\\[2pt]
  x_{p,i} &\;\in\; \{0,1\},
    && \forall\, p,i = 1,\ldots,P. \tag{1c}
\end{align}$$

#### Scheduling constraints (2).

$$\begin{align}
  t_1 &\;=\; 0, \tag{2a}\\[2pt]
  0 \;\le\; t_i &\;\le\; L,
    && \forall\, i = 1,\ldots,P, \tag{2b}\\[2pt]
  t_i &\;\ge\; t_{i-1},
    && \forall\, i = 2,\ldots,P. \tag{2c}
\end{align}$$

## Second-Stage Recourse $Q(x,t,\xi)$

Given first-stage $(x,t)$ and realization $\xi$, let $Q(x,t,\xi)$ denote the operational cost (patient waiting + provider idle + provider overtime) as defined in the paper; it is the optimal value of the second-stage LP that determines waiting, idle, and overtime quantities.

## DRO Objective (Equation 6)

$$\begin{equation}
  \min_{(x,t)\,\in\,\mathcal{X}\times\mathcal{T}} \;
    \sup_{P \,\in\, \mathcal{F}(S,\mu)} \;
      \mathbb{E}_P\!\bigl[\, Q(x,t,\xi) \,\bigr],
  \tag{6}
\end{equation}$$ where $\mathcal{X}$ is the set of $x$ satisfying (1a)–(1c) and $\mathcal{T}$ is the set of $t$ satisfying (2a)–(2c).

**Original (non-reformulated) structure.** The outer DRO problem is a min–max over the infinite-dimensional ambiguity set $\mathcal{F}(S,\mu)$. The tractable MILP obtained via Proposition 4 / Eq. (15) is *not* substituted here.
