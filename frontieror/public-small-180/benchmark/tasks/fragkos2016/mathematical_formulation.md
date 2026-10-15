# Original Formulation: Capacitated Lot-Sizing Problem with Setup Times (CLST)

*Source: A Horizon Decomposition Approach for the Capacitated Lot-Sizing Problem with Setup Times, Ioannis Fragkos, Zeger Degraeve, Bert De Reyck, 2016 (INFORMS Journal on Computing 28(3): 465–482). Section 3.1, p. 467.*

## Sets and Indices

$$\begin{align*}
&I = \{1, \dots, n\}: && \text{set of items, indexed by } i.\\
&T = \{1, \dots, m\}: && \text{set of periods, indexed by } t.
\end{align*}$$

## Parameters

$$\begin{align*}
&d_{it}: && \text{demand of item } i \text{ in period } t, \quad i \in I,\ t \in T.\\
&sd_{itk}: && \text{sum of demand of item } i \text{ from period } t \text{ to period } k, \quad i \in I,\ t \in T,\ k \in T:\ t \le k.\\
&hc_{it}: && \text{cost of holding inventory for item } i \text{ from period } t-1 \text{ to period } t, \quad i \in I,\ t \in T.\\
&sc_{it}: && \text{setup cost of item } i \text{ in period } t, \quad i \in I,\ t \in T.\\
&vc_{it}: && \text{production cost of item } i \text{ in period } t, \quad i \in I,\ t \in T.\\
&st_{it}: && \text{setup time of item } i \text{ in period } t, \quad i \in I,\ t \in T.\\
&vt_{it}: && \text{variable production time of item } i \text{ in period } t, \quad i \in I,\ t \in T.\\
&cap_{t}: && \text{time capacity in period } t, \quad t \in T.\\
&M_{it}: && \text{big-M quantity, defined as } M_{it} = \min\{\, sd_{itm},\ (cap_t - st_{it})/(vt_{it}) \,\}, \quad i \in I,\ t \in T.
\end{align*}$$

## Decision Variables

$$\begin{align*}
&x_{it}: && \text{production quantity of item } i \text{ in period } t, \quad i \in I,\ t \in T.\\
&s_{it}: && \text{inventory quantity of item } i \text{ at the beginning of period } t, \quad i \in I,\ t \in T \cup \{m+1\}.\\
&y_{it}: && \text{equals } 1 \text{ if a setup occurs for item } i \text{ in period } t,\ 0 \text{ otherwise}, \quad i \in I,\ t \in T.
\end{align*}$$

## Objective

$$\begin{equation}
\min \quad \left\{ \sum_{i \in I} \sum_{t \in T} sc_{it}\, y_{it} \;+\; \sum_{i \in I} \sum_{t \in T} vc_{it}\, x_{it} \;+\; \sum_{i \in I} \sum_{t \in T} hc_{it}\, s_{it} \right\} \tag{1}
\end{equation}$$

## Constraints

$$\begin{align}
& s_{it} + x_{it} = d_{it} + s_{i,t+1} && \forall i \in I,\ \forall t \in T \tag{2}\\[4pt]
& x_{it} \le M_{it}\, y_{it} && \forall i \in I,\ \forall t \in T \tag{3}\\[4pt]
& \sum_{i \in I} st_{it}\, y_{it} + \sum_{i \in I} vt_{it}\, x_{it} \le cap_t && \forall t \in T \tag{4}\\[4pt]
& x_{it},\, s_{it} \ge 0,\quad s_{i,m+1} = 0,\quad y_{it} \in \{0,1\} && \forall i \in I,\ \forall t \in T \tag{5}
\end{align}$$
