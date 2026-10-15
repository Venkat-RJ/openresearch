# Sets and Parameters

- $J = \{1,\ldots,|J|\}$: set of potential facility locations; index $j$.

- $I = \{1,\ldots,|I|\}$: set of customers; index $i$.

- $d_i$: demand rate of customer $i \in I$.

- $f_j$: fixed setup cost to open facility $j \in J$.

- $c_{ij}$: unit shipment cost from facility $j \in J$ to customer $i \in I$.

- $c_{i0}$: unit penalty cost if customer $i$ is not served ($c_{i0} \geq c_{ij}$ for all $i,j$).

- $\tilde{\boldsymbol{\xi}} = (\tilde{\xi}_1,\ldots,\tilde{\xi}_{|J|})^\top$: random binary disruption vector with $\tilde{\xi}_j = 1$ if facility $j$ is online, $0$ if disrupted.

- $\Xi := \{ (\xi_1,\ldots,\xi_{|J|})^\top \mid \xi_j \in \{0,1\},\; \forall j \in J \} = \{0,1\}^{|J|}$.

- Dummy facility $0$: $x_0 \equiv 1$, $\xi_0 \equiv 1$.

- For $k \in \{1,\ldots,n\}$: $A_k, B_k \subseteq J$ are the sets of online and offline facilities described by the $k$-th piece of information, associated with probability interval $[\underline{q}_k, \overline{q}_k]$.

# Ambiguity Set (Equation (1))

For any realization $\boldsymbol{\xi} \in \Xi$, let $p_{\boldsymbol{\xi}} := \Pr(\tilde{\boldsymbol{\xi}} = \boldsymbol{\xi})$. The ambiguity set of admissible distributions $\boldsymbol{p} = (p_{\boldsymbol{\xi}})_{\boldsymbol{\xi} \in \Xi}$ is $$\begin{equation}
P := \left\{\, \boldsymbol{p} \in [0,1]^{2^{|J|}} \;\left|\;
\begin{aligned}
& \sum_{\substack{\boldsymbol{\xi} \in \Xi \,:\, \xi_j = 1 \;\forall j \in A_k, \\ \xi_j = 0 \;\forall j \in B_k}} p_{\boldsymbol{\xi}} \in [\underline{q}_k, \overline{q}_k] \quad \forall k \in \{1,\ldots,n\}, \\[2pt]
& \sum_{\boldsymbol{\xi} \in \Xi} p_{\boldsymbol{\xi}} = 1
\end{aligned}
\right.\right\}. \tag{1}
\end{equation}$$

# Second-Stage Transportation and Penalty Cost

Given a facility location decision $\boldsymbol{x}$ and a disruption realization $\boldsymbol{\xi}$, $$\begin{equation}
Q(\boldsymbol{x}, \boldsymbol{\xi}) := \sum_{i \in I} d_i \cdot \min_{j \in J \cup \{0\} \,:\, x_j \xi_j = 1} c_{ij},
\end{equation}$$ with the convention $x_0 \equiv 1$ and $\xi_0 \equiv 1$ (the penalty cost $c_{i0}$ is incurred only when all open facilities are disrupted).

# Decision Variables

$$\begin{align*}
x_j &\in \{0,1\} && \forall\, j \in J \quad (\text{1 if facility $j$ is opened, 0 otherwise}) \\
\boldsymbol{x} &= (x_1, x_2, \ldots, x_{|J|})^\top
\end{align*}$$

# Objective and Problem $\mathcal{P}$ (Equation (3))

The two-stage distributionally robust formulation minimizes the fixed setup cost plus the worst-case expected transportation and penalty cost over all distributions in the ambiguity set $P$: $$\begin{equation}
\mathcal{P}: \quad \min_{\boldsymbol{x} \in \{0,1\}^{|J|}} \left\{\, \sum_{j \in J} f_j\, x_j \;+\; \max_{\boldsymbol{p} \in P} \; \mathbb{E}_{\boldsymbol{p}}\!\left[\, Q(\boldsymbol{x}, \tilde{\boldsymbol{\xi}}) \,\right] \right\}. \tag{3}
\end{equation}$$

# Constraints

$$\begin{align*}
x_j &\in \{0,1\} && \forall\, j \in J.
\end{align*}$$ (All additional structure is captured implicitly by the definition of $Q$ and by the ambiguity set $P$.)
