# Sets and Indices

- $\mathcal{P}$: set of products, indexed by $p$.

- $\mathcal{F}$: set of facilities, indexed by $f$.

- $\mathcal{L}_{pf}$: set of potential production levels for product $p$ at facility $f$, indexed by $l$.

- $\mathcal{D}_p$: set of joint distributions of yield and demand for product $p$, indexed by $d$.

- $\mathcal{S}_{pd}$: set of scenarios under distribution $d \in \mathcal{D}_p$ for product $p$, indexed by $s$.

# Parameters

- $L_{pfl}, U_{pfl}$: lower/upper production-volume limits of level $l$ for product $p$ at facility $f$.

- $B_f$: capacity of facility $f$.

- $C_{pf}$: unit allocation cost of product $p$ at facility $f$ (with $O_p < C_{pf}$).

- $P_p$: sale price per unit of product $p$; $O_p$: discounted price per unit of unsold inventory of product $p$.

- $\pi_{sd}$: probability of scenario $s$ under distribution $d$.

- $Y_{pfds}$: yield realization for product $p$ at facility $f$ in scenario $s$ of distribution $d$.

- $D_{pds}$: demand realization for product $p$ in scenario $s$ of distribution $d$.

- $l(p,f,d)$: production level at facility $f$ for product $p$ that must be in effect under distribution $d$.

# Decision Variables

**First stage:**

- $x_{pf} \geq 0$: raw-material allocation of product $p$ to facility $f$.

- $y_{pfl} \in \{0,1\}$: $=1$ iff level $l$ is chosen for product $p$ at facility $f$.

- $\delta_{pd} \in \{0,1\}$: $=1$ iff distribution $d$ describes yield/demand for product $p$.

**Second stage:**

- $z_{pds} \geq 0$: available inventory of product $p$ in scenario $s$ of distribution $d$.

- $w_{pds} \geq 0$: amount of product $p$ sold at full price in scenario $s$ of distribution $d$.

- $o_{pds} \geq 0$: amount of product $p$ sold at discounted price in scenario $s$ of distribution $d$.

# Original (Bilinear) Formulation

$$\begin{align}
v^{*}_{\text{PP-DESUP}} = \max\quad
  & -\sum_{p \in \mathcal{P}} \sum_{f \in \mathcal{F}} C_{pf}\, x_{pf}
    + \sum_{p \in \mathcal{P}} \sum_{d \in \mathcal{D}_p} \delta_{pd}
      \left[\, \sum_{s \in \mathcal{S}_{pd}} \pi_{sd}\,
             \bigl(P_p\, w_{pds} + O_p\, o_{pds}\bigr) \right]
  \tag{1}
\end{align}$$

subject to

$$\begin{align}
  \sum_{p \in \mathcal{P}} x_{pf} &\leq B_f,
    &&\forall f \in \mathcal{F}, \tag{2}\\
  \sum_{l \in \mathcal{L}_{pf}} y_{pfl} &= 1,
    &&\forall p \in \mathcal{P},\, f \in \mathcal{F}, \tag{3}\\
  \sum_{l \in \mathcal{L}_{pf}} L_{pfl}\, y_{pfl}
    \;\leq\; x_{pf}
    &\;\leq\; \sum_{l \in \mathcal{L}_{pf}} U_{pfl}\, y_{pfl},
    &&\forall p \in \mathcal{P},\, f \in \mathcal{F}, \tag{4}\\
  \sum_{d \in \mathcal{D}_p} \delta_{pd} &= 1,
    &&\forall p \in \mathcal{P}, \tag{5}\\
  \sum_{f \in \mathcal{F}} y_{p,f,l(p,f,d)} &\geq |\mathcal{F}|\, \delta_{pd},
    &&\forall p \in \mathcal{P},\, d \in \mathcal{D}_p, \tag{6}\\
  z_{pds} &= \sum_{f \in \mathcal{F}} Y_{pfds}\, x_{pf},
    &&\forall p \in \mathcal{P},\, d \in \mathcal{D}_p,\, s \in \mathcal{S}_{pd}, \tag{7}\\
  w_{pds} &\leq D_{pds},
    &&\forall p \in \mathcal{P},\, d \in \mathcal{D}_p,\, s \in \mathcal{S}_{pd}, \tag{8}\\
  w_{pds} + o_{pds} &= z_{pds},
    &&\forall p \in \mathcal{P},\, d \in \mathcal{D}_p,\, s \in \mathcal{S}_{pd}, \tag{9}\\
  x_{pf} &\geq 0,
    &&\forall p \in \mathcal{P},\, f \in \mathcal{F}, \tag{10}\\
  y_{pfl} &\in \{0,1\},
    &&\forall p \in \mathcal{P},\, f \in \mathcal{F},\, l \in \mathcal{L}_{pf}, \tag{11}\\
  \delta_{pd} &\in \{0,1\},
    &&\forall p \in \mathcal{P},\, d \in \mathcal{D}_p, \tag{12}\\
  z_{pds},\; w_{pds},\; o_{pds} &\geq 0,
    &&\forall p \in \mathcal{P},\, d \in \mathcal{D}_p,\, s \in \mathcal{S}_{pd}. \tag{13}
\end{align}$$

The objective (1) contains the bilinear terms $\delta_{pd}\,w_{pds}$ and $\delta_{pd}\,o_{pds}$; this is the original (non-linearized) form.
