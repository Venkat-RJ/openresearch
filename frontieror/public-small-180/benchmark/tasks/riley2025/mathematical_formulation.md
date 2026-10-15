# Original Formulation (Case Study 1, Eqs. 11–24)

## Sets and Indices

- $\mathcal{I}$: set of demand sites, indexed by $i$.

- $\mathcal{J}$: set of production sites, indexed by $j$ (also $j' \in \mathcal{J}$).

- $\mathcal{M}$: set of module sizes, indexed by $m$.

- $\mathcal{T}$: set of time periods, indexed by $t$.

- $\mathcal{S}$: set of scenarios, indexed by $s$.

- $\mathcal{K}_m$: set of investment levels for module size $m$, indexed by $k$.

- $\mathcal{K}_1 \subseteq \mathcal{K}_m$: first investment level per size.

## Parameters

- $p_s$: probability of scenario $s$.

- $c_{ijt}$: unit cost of satisfying demand at site $i$ from site $j$ in period $t$.

- $d_{its}$: demand at site $i$ in period $t$ under scenario $s$.

- $h_{jj'mt}$: unit cost of relocating a module of size $m$ from site $j$ to $j'$ in period $t$.

- $p_{it}$: unit penalty for unmet demand at site $i$ in period $t$.

- $g_{mk}$: capital cost for selecting manufacturing unit $k$ of size $m$.

- $u_m$: production capacity of a single unit of size $m$.

- $u^{\max}$: maximum total production at any site $j$ in any period.

## Decision Variables

First-stage (binary investment decisions):

- $y_{mk} \in \{0,1\}$: 1 if manufacturing unit $k$ of size $m$ is selected.

Second-stage (scenario-dependent operational decisions):

- $z_{ms} \in \mathbb{Z}_+$: number of units of size $m$ (selected) in scenario $s$.

- $x_{ijts} \in [0,1]$: fraction of demand at site $i$ satisfied from site $j$ in period $t$, scenario $s$.

- $q_{its} \in [0,1]$: fraction of demand at site $i$ not delivered in period $t$, scenario $s$.

- $v_{jmts} \in \mathbb{Z}_+$: number of units of size $m$ at site $j$ at time $t$, scenario $s$.

- $w_{jj'mts} \in \mathbb{Z}_+$: number of units of size $m$ relocated from $j$ to $j'$ at time $t$, scenario $s$.

## Objective and Constraints

$$\begin{align}
\min \quad & \sum_{s \in \mathcal{S}} p_s \Bigg( \sum_{j \in \mathcal{J}} \sum_{t \in \mathcal{T}} \bigg( \sum_{i \in \mathcal{I}} c_{ijt} d_{its} x_{ijts} + \sum_{j' \in \mathcal{J}, m \in \mathcal{M}} h_{jj'mt} w_{jj'mts} \bigg) \notag \\
& \qquad\qquad + \sum_{i \in \mathcal{I}, t \in \mathcal{T}} p_{it} q_{its} d_{its} \Bigg) + \sum_{m \in \mathcal{M}} \sum_{k \in \mathcal{K}_m} g_{mk} y_{mk} \tag{11} \\
\text{s.t.}\quad & y_{mk} \leq y_{m,k-1} && \forall m \in \mathcal{M},\ k = 2,\ldots,|\mathcal{K}_m| \tag{12} \\
& z_{ms} = \sum_{k \in \mathcal{K}_m} y_{mk} && \forall m \in \mathcal{M},\ s \in \mathcal{S} \tag{13} \\
& \sum_{j \in \mathcal{J}} x_{ijts} + q_{its} = 1 && \forall i \in \mathcal{I},\ t \in \mathcal{T},\ s \in \mathcal{S} \tag{14} \\
& v_{0mts} = z_{ms} + \sum_{t'=1}^{t} \bigg( \sum_{j' \in \mathcal{J}} w_{j'0mts'} - \sum_{j' \in \mathcal{J}} w_{0j'mts'} \bigg) && \forall m \in \mathcal{M},\ t \in \mathcal{T},\ s \in \mathcal{S} \tag{15} \\
& v_{jmts} = \sum_{t'=1}^{t} \bigg( \sum_{j' \in \mathcal{J}} w_{j'jmts'} - \sum_{j' \in \mathcal{J}} w_{jj'mts'} \bigg) && \forall j \in \mathcal{J},\ m \in \mathcal{M},\ t \in \mathcal{T},\ s \in \mathcal{S} \tag{16} \\
& \sum_{i \in \mathcal{I}} d_{its} x_{ijts} \leq \sum_{m \in \mathcal{M}} v_{jmts} u_m && \forall j \in \mathcal{J},\ t \in \mathcal{T},\ s \in \mathcal{S} \tag{17} \\
& \sum_{i \in \mathcal{I}} d_{its} x_{ijts} \leq u^{\max} && \forall j \in \mathcal{J},\ t \in \mathcal{T},\ s \in \mathcal{S} \tag{18} \\
& y_{mk} \in \{0, 1\} && \forall m \in \mathcal{M},\ k \in \mathcal{K}_m \tag{19} \\
& 0 \leq q_{its} \leq 1 && \forall i \in \mathcal{I},\ t \in \mathcal{T},\ s \in \mathcal{S} \tag{20} \\
& 0 \leq x_{ijts} \leq 1 && \forall i \in \mathcal{I},\ j \in \mathcal{J},\ t \in \mathcal{T},\ s \in \mathcal{S} \tag{21} \\
& z_{jms} \in \mathbb{Z}_+ && \forall j \in \mathcal{J},\ m \in \mathcal{M},\ s \in \mathcal{S} \tag{22} \\
& v_{jmts} \in \mathbb{Z}_+ && \forall j \in \mathcal{J},\ m \in \mathcal{M},\ t \in \mathcal{T},\ s \in \mathcal{S} \tag{23} \\
& w_{jj'mts} \in \mathbb{Z}_+ && \forall j \in \mathcal{J},\ j' \in \mathcal{J},\ m \in \mathcal{M},\ t \in \mathcal{T},\ s \in \mathcal{S} \tag{24}
\end{align}$$
