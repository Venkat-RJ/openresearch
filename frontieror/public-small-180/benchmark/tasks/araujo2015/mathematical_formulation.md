# Original Formulation: Capacitated Lot Sizing Problem with Setup Times (CLST)

*Source: Period Decompositions for the Capacitated Lot Sizing Problem with Setup Times, Silvio Alexandre de Araujo, Bert De Reyck, Zeger Degraeve, Ioannis Fragkos, Raf Jans, 2015.*

This is the *regular* formulation (CL) presented in Section 3.1, which defines the problem the paper studies and whose notation ($x_{it}, y_{it}, s_{it}$) is carried into all subsequent reformulations (SP, SPt, FL, FLp), decompositions, and experiments.

## Sets and Indices

- $I$: set of items, $= \{1, \dots, |I|\}$, indexed by $i$.

- $T$: set of time periods, $= \{1, \dots, |T|\}$, indexed by $t$ (and $k$).

## Parameters

- $d_{it}$: demand of item $i$ in period $t$, $\forall i \in I,\ \forall t \in T$.

- $sd_{itk}$: sum of demand of item $i$, from period $t$ until $k$, $\forall i \in I,\ \forall t \in T,\ \forall k \in T: k \geq t$ (i.e. $sd_{itk} = \sum_{u=t}^{k} d_{iu}$).

- $hc_{it}$: unit holding cost for item $i$ in period $t$, $\forall i \in I,\ \forall t \in T$.

- $sc_{it}$: setup cost for item $i$ in period $t$, $\forall i \in I,\ \forall t \in T$.

- $vc_{it}$: variable production cost for item $i$ in period $t$, $\forall i \in I,\ \forall t \in T$.

- $fc_{i}$: unit cost for initial inventory for item $i$, $\forall i \in I$.

- $st_{it}$: setup time for item $i$ in period $t$, $\forall i \in I,\ \forall t \in T$.

- $vt_{it}$: variable production time for item $i$ in period $t$, $\forall i \in I,\ \forall t \in T$.

- $cap_{t}$: time capacity in period $t$, $\forall t \in T$.

## Decision Variables

- $x_{it}$: production quantity of item $i$ in period $t$, $\forall i \in I,\ \forall t \in T$.

- $y_{it}$: $= 1$ if setup for item $i$ in period $t$, $0$ otherwise, $\forall i \in I,\ \forall t \in T$.

- $s_{it}$: inventory for item $i$ at the end of period $t$, $\forall i \in I,\ \forall t \in T$.

- $s_{i0}$: amount of initial inventory for item $i$, $\forall i \in I$.

## Objective

$$\begin{equation}
\min \left\{ \sum_{i \in I} fc_i\, s_{i0}
  + \sum_{i \in I} \sum_{t \in T} \left( sc_{it}\, y_{it} + vc_{it}\, x_{it} + hc_{it}\, s_{it} \right) \right\}
\tag{1}
\end{equation}$$

## Constraints

$$\begin{align}
& s_{i,t-1} + x_{it} = d_{it} + s_{it}
  && \forall\, i \in I,\ \forall\, t \in T \tag{2} \\[4pt]
& \sum_{i \in I} \left( st_{it}\, y_{it} + vt_{it}\, x_{it} \right) \leq cap_t
  && \forall\, t \in T \tag{3} \\[4pt]
& x_{it} \leq \min\!\left( \frac{cap_t - st_{it}}{vt_{it}},\; sd_{it|T|} \right) y_{it}
  && \forall\, i \in I,\ \forall\, t \in T \tag{4} \\[4pt]
& y_{it} \in \{0,1\},\ x_{it} \geq 0,\ s_{it} \geq 0,\ s_{i0} \geq 0,\ s_{i|T|} = 0
  && \forall\, i \in I,\ \forall\, t \in T \tag{5}
\end{align}$$
