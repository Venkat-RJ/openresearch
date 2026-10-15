# Original Formulation: Integrated Lot Sizing and Scheduling Problem with Non-Triangular Setup on Parallel Machines (ILSSP-NT)

*Source: Hybrid matheuristics to solve the integrated lot sizing and scheduling problem on parallel machines with sequence-dependent and non-triangular setup, Desiree M. Carvalho, Mariá C. V. Nascimento, 2022.*

## Sets and Indices

$$\begin{align*}
&n && \text{number of items} \\
&m && \text{number of machines} \\
&p && \text{number of periods (macro-periods)} \\
&i, j \in \{1,\ldots,n\} && \text{indexes representing items} \\
&k \in \{1,\ldots,m\}    && \text{index representing the machines} \\
&t, u \in \{1,\ldots,p\} && \text{indexes representing the macro-periods}
\end{align*}$$

## Parameters

$$\begin{align*}
&c_{ijk} && \text{setup cost to change the state of machine } k \text{ from item } i \text{ to item } j \\
&h_i     && \text{unitary inventory cost of item } i \\
&d_{it}  && \text{demand of item } i \text{ in period } t \\
&b_{ijk} && \text{setup time to change the state of machine } k \text{ from item } i \text{ to item } j \\
&M_{it}  && \text{maximum amount of item } i \text{ that can be produced in period } t \\
&f_{ikt} && \text{processing time of item } i \text{ at machine } k \text{ in period } t \\
&m_i     && \text{minimum production lot size of item } i \\
&T_{kt}  && \text{production capacity of machine } k \text{ in period } t \\
&q_{ikt} && \text{maximum number of times machine } k \text{ can be set up to produce item } i \text{ in period } t
\end{align*}$$

## Decision Variables

$$\begin{align*}
&x_{iktu}  && \text{amount of item } i \text{ produced at machine } k \text{ in period } t \text{ to meet the demand of period } u \;(\geq 0) \\
&x^b_{ikt} && \text{amount of item } i \text{ produced at machine } k \text{ in the beginning of period } t, \\
&         && \text{before the first machine setup is performed in period } t \;(\geq 0) \\
&x^a_{ikt} && \text{amount of item } i \text{ produced at machine } k \text{ during period } t, \\
&         && \text{after the first machine setup is performed} \;(\geq 0) \\
&z_{ikt}   && \text{binary; } 1 \text{ if machine } k \text{ is ready to produce item } i \text{ at the beginning of period } t \text{ (setup carry-over)} \\
&y_{ijkt}  && \text{number of times a production process changes from item } i \text{ to item } j \text{ at machine } k \text{ in period } t \\
&R_{kt}    && \text{binary; } 1 \text{ if at least one setup is performed at machine } k \text{ in period } t \;(\textstyle\sum_{i=1}^{n}\sum_{j=1}^{n} y_{ijkt}\geq 1) \\
&G_{ikt}   && \text{binary; } 1 \text{ if machine } k \text{ is ready at least once to produce item } i \text{ during period } t \\
&F_{ijkt}  && \text{commodity flow from node (item) } i \text{ to node (item) } j \text{ at machine } k \text{ in period } t \;(\geq 0)
\end{align*}$$

## Objective

$$\begin{align}
\min \quad \sum_{i=1}^{n}\sum_{k=1}^{m}\sum_{t=1}^{p}\sum_{u=t}^{p}(u-t)\,h_i\,x_{iktu}
   \;+\; \sum_{i=1}^{n}\sum_{j=1}^{n}\sum_{k=1}^{m}\sum_{t=1}^{p} c_{ijk}\,y_{ijkt}
   \tag{1}
\end{align}$$

## Constraints

$$\begin{align}
& \sum_{k}\sum_{t=1}^{u} x_{iktu} = d_{iu}
   && \forall (i,u) \tag{2} \\[2pt]
& \sum_{i}\left( \sum_{u=t}^{p} f_{ikt}\,x_{iktu} + \sum_{j} b_{jik}\,y_{jikt} \right) \leq T_{kt}
   && \forall (k,t) \tag{3} \\[2pt]
& x_{iktu} \leq M_{it}\,G_{ikt}
   && \forall (i,k,t,u) \tag{4} \\[2pt]
& \sum_{i} z_{ikt} = 1
   && \forall (k,t) \tag{5} \\[2pt]
& z_{ikt} + \sum_{j} y_{jikt} = \sum_{j} y_{ijkt} + z_{ik,t+1}
   && \forall (i,k,t) \tag{6} \\[2pt]
& z_{ikt} + \sum_{j} y_{jikt} \geq G_{ikt}
   && \forall (i,k,t) \tag{7} \\[2pt]
& z_{ikt} + \sum_{j} y_{jikt} \leq q_{ikt}\,G_{ikt}
   && \forall (i,k,t) \tag{8} \\[2pt]
& \sum_{u=t}^{p} x_{iktu} = x^a_{ikt} + x^b_{ikt}
   && \forall (i,k,t) \tag{9} \\[2pt]
& x^b_{ikt} \leq M_{it}\,z_{ikt}
   && \forall (i,k,t) \tag{10} \\[2pt]
& x^a_{ikt} \geq m_i\left( \sum_{j} y_{jikt} - z_{ik,t+1} \right)
   && \forall (i,j,k,t) \tag{11} \\[2pt]
& x^a_{ikt} + \sum_{\lambda=t+1}^{u} x^b_{ik\lambda} \geq m_i \sum_{j} y_{jikt}
   - M_{it}\left( \sum_{\lambda=t+1}^{u-1} R_{k\lambda} + 1 - R_{ku} \right)
   && \forall (i,k,t,u),\, u \neq t \tag{12} \\[2pt]
& \sum_{j} F_{0jkt} = \sum_{i} G_{ikt}
   && \forall (k,t) \tag{13} \\[2pt]
& F_{0ikt} + \sum_{j} F_{jikt} = G_{ikt} + \sum_{j} F_{ijkt}
   && \forall (i,k,t) \tag{14} \\[2pt]
& F_{0ikt} \leq n\,z_{ikt}
   && \forall (i,k,t) \tag{15} \\[2pt]
& F_{ijkt} \leq n\,y_{ijkt}
   && \forall (i,j,k,t) \tag{16} \\[2pt]
& F_{ijkt} \geq 0
   && \forall (i,j,k,t) \tag{17} \\[2pt]
& G_{ikt},\, R_{kt} \in \{0,1\}
   && \forall (i,k,t) \tag{18} \\[2pt]
& z_{ikt} \in \{0,1\},\; x_{iktu} \geq 0
   && \forall (i,k,t,u) \tag{19} \\[2pt]
& y_{ijkt} \in \{0,\ldots,q_{jkt}\}
   && \forall (i,j,k,t) \tag{20}
\end{align}$$
