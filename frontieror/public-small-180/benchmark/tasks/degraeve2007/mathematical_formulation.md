# Original Formulation: Capacitated Lot Sizing Problem with Set Up Times (CLST)

*Source: A New Dantzig-Wolfe Reformulation and Branch-and-Price Algorithm for the Capacitated Lot Sizing Problem with Set Up Times, Zeger Degraeve and Raf Jans, 2003/2007.*

## Sets and Indices

$$\begin{align*}
P &: \text{set of products},\ = \{1,\dots,n\}, \text{ indexed by } i. \\
T &: \text{set of time periods},\ = \{1,\dots,m\}, \text{ indexed by } t.
\end{align*}$$

## Parameters

$$\begin{align*}
d_{it}    &: \text{demand of product } i \text{ in period } t, &&\forall i \in P,\ \forall t \in T \\
sd_{itk}  &: \text{sum of demand of product } i, \text{ from period } t \text{ until period } k, &&\forall i \in P,\ \forall t,k \in T : k \geq t \\
hc_{it}   &: \text{holding cost for product } i \text{ in period } t, &&\forall i \in P,\ \forall t \in T \\
sc_{it}   &: \text{set up cost for product } i \text{ in period } t, &&\forall i \in P,\ \forall t \in T \\
vc_{it}   &: \text{variable production cost for product } i \text{ in period } t, &&\forall i \in P,\ \forall t \in T \\
fc_{i}    &: \text{unit cost for initial inventory for product } i, &&\forall i \in P \\
st_{it}   &: \text{set up time for product } i \text{ in period } t, &&\forall i \in P,\ \forall t \in T \\
vt_{it}   &: \text{variable production time for product } i \text{ in period } t, &&\forall i \in P,\ \forall t \in T \\
cap_{t}   &: \text{capacity in period } t, &&\forall t \in T
\end{align*}$$

## Decision Variables

$$\begin{align*}
x_{it} &: \text{production of product } i \text{ in period } t, &&\forall i \in P,\ \forall t \in T \\
y_{it} &: = 1 \text{ if set up for product } i \text{ in period } t,\ = 0 \text{ otherwise}, &&\forall i \in P,\ \forall t \in T \\
si_{i} &: \text{amount of initial inventory for item } i, &&\forall i \in P \\
s_{it} &: \text{inventory of product } i \text{ at the end of period } t, &&\forall i \in P,\ \forall t \in T
\end{align*}$$

## Objective

$$\begin{equation}
\min\ \sum_{i \in P} fc_i\, si_i + \sum_{i \in P} \sum_{t \in T} \left( sc_{it}\, y_{it} + vc_{it}\, x_{it} + hc_{it}\, s_{it} \right) \tag{1}
\end{equation}$$

## Constraints

$$\begin{align}
si_i + x_{i,1} &= d_{i,1} + s_{i,1} && \forall i \in P \tag{2.1} \\
s_{i,t-1} + x_{it} &= d_{it} + s_{it} && \forall i \in P,\ \forall t \in T \setminus \{1\} \tag{2.2} \\
x_{it} &\leq \min\!\left\{ (cap_t - st_{it})/vt_{it},\ sd_{itm} \right\} y_{it} && \forall i \in P,\ \forall t \in T \tag{3} \\
\sum_{i \in P} \left( st_{it}\, y_{it} + vt_{it}\, x_{it} \right) &\leq cap_t && \forall t \in T \tag{4} \\
y_{it} \in \{0,1\},\quad x_{it} \geq 0,\quad s_{it} &\geq 0 && \forall i \in P,\ \forall t \in T \tag{5}
\end{align}$$
