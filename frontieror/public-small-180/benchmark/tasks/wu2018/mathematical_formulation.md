# Original Formulation: Capacitated Multi-Item Lot Sizing Problem with Nonidentical Machines (CMLS-NM)

*Source: Analytics Branching and Selection for the Capacitated Multi-Item Lot Sizing Problem with Nonidentical Machines, Tao Wu, Zhe Liang, Canrong Zhang, 2018 (INFORMS Journal on Computing 30(2):236–258).*

The paper proposes three equivalent formulations: the Lot Sizing formulation with Nonidentical Machines (LSNM, §3.1), the Facility Location formulation (FLNM, §3.2), and the Shortest Path formulation (SPNM, §3.3). LSNM is the formulation written first as the definition of the problem and whose notation $(X,I,Y)$ is carried throughout; it is the canonical original formulation reproduced below. FLNM and SPNM are listed under *Variants*.

## Sets and Indices

- $T$ — set of periods in the planning horizon, $T=\{1,\dots,|T|\}$, indexed by $t$, $s$, and $q$.

- $J$ — set of items, $J=\{1,\dots,|J|\}$, indexed by $j$.

- $M$ — set of machines, $M=\{1,\dots,|M|\}$, indexed by $m$.

- $\mathcal{M}_j$ — set of machines that are capable of producing item $j$.

- $\mathcal{F}_m$ — set of items that can be produced on machine $m$.

## Parameters

- $d_{jt}$ — demand for item $j$ in period $t$.

- $sd_{jts}$ — sum of demand for item $j$ from period $t$ until $s$ ($s\geq t$).

- $sc^m_j$ — setup cost for producing a lot of item $j$ on machine $m$ (period independent).

- $st^m_j$ — setup time for producing a lot of item $j$ on machine $m$ (period independent).

- $hc_j$ — inventory-holding cost for one unit of item $j$ for one period.

- $I_{j,0}$ — initial inventory of item $j$ at the start of the planning horizon (given, $I_{j,0}\geq 0$).

- $pc^m_j$ — variable cost of producing one unit of item $j$ on machine $m$ (period independent).

- $pt^m_j$ — time required to produce one unit of item $j$ on machine $m$ (period independent).

- $C^m_t$ — available capacity of machine $m$ in period $t$.

- $BM^m_{jt}$ — big values reflecting machine capacities and the required production amount in an optimal plan: $BM^m_{jt}=\min\!\Big(\sum_{q\in\{t,\dots,|T|\}} d_{jq},\;(C^m_t-st^m_j)/pt^m_j\Big)$.

- $mm_j$ — maximum number of machines allowed to produce item $j$ in each period; we require $mm_j\leq|\mathcal{M}_j|$.

## Decision Variables

- $X^m_{jt}$ — number of units of item $j$ produced on machine $m$ in period $t$ (continuous, $\geq 0$).

- $I_{jt}$ — inventory of item $j$ at the end of period $t$ (continuous, $\geq 0$).

- $Y^m_{jt}$ — setup variable: $Y^m_{jt}=1$ if $X^m_{jt}>0$, and $0$ otherwise (binary).

## Objective

$$\begin{equation}
\min\left\{ \sum_{j\in J}\sum_{t\in T}\sum_{m\in\mathcal{M}_j} sc^m_j\cdot Y^m_{jt}
      + \sum_{j\in J}\sum_{t\in T}\sum_{m\in\mathcal{M}_j} pc^m_j\cdot X^m_{jt}
      + \sum_{j\in J}\sum_{t\in T} hc_j\cdot I_{jt} \right\}
\tag{1}
\end{equation}$$

## Constraints

$$\begin{align}
& \sum_{m\in\mathcal{M}_j} X^m_{jt} + I_{j(t-1)} - I_{jt} = d_{jt},
   && \forall\, j\in J,\; t\in T; \tag{2}\\[4pt]
& \sum_{j\in\mathcal{F}_m} pt^m_j\cdot X^m_{jt} + \sum_{j\in\mathcal{F}_m} st^m_j\cdot Y^m_{jt} \leq C^m_t,
   && \forall\, t\in T,\; m\in M; \tag{3}\\[4pt]
& X^m_{jt} \leq BM^m_{jt}\cdot Y^m_{jt},
   && \forall\, j\in J,\; t\in T,\; m\in\mathcal{M}_j; \tag{4}\\[4pt]
& \sum_{m\in\mathcal{M}_j} Y^m_{jt} \leq mm_j,
   && \forall\, j\in J,\; t\in T; \tag{5}\\[4pt]
& X^m_{jt},\; I_{jt} \geq 0,
   && \forall\, j\in J,\; t\in T,\; m\in\mathcal{M}_j; \tag{6}\\[4pt]
& Y^m_{jt} \in \{0,1\},
   && \forall\, j\in J,\; t\in T,\; m\in\mathcal{M}_j. \tag{7}
\end{align}$$
