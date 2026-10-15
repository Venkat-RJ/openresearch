# Original Formulation: Stochastic Production Routing Problem (SPRP)

*Source: Benders Decomposition for Production Routing Under Demand Uncertainty, Yossiri Adulyasak, Jean-François Cordeau, Raf Jans, Operations Research, 2015.*

The paper studies the Stochastic Production Routing Problem (SPRP). The canonical model below is the *two-stage SPRP formulation (2-BF)*, Model (1)–(19), introduced first in §2.2 as the definition of the problem; its notation $(x,y,z,p,I,q,e)$ is carried into every subsequent section. The multistage variant (M-BF, adding nonanticipativity constraints (20)–(23)) is given under Variants.

## Sets and Indices

$$\begin{align*}
&\Omega && \text{finite set of demand scenarios, indexed by } \omega\\
&G=(N,E) && \text{complete undirected graph}\\
&N=\{0,\ldots,n\} && \text{set of nodes; node } 0 \text{ is the production plant}\\
&N_c=N\setminus\{0\} && \text{set of customers}\\
&E=\{(i,j): i,j\in N,\ i<j\} && \text{set of edges}\\
&E(S) && \{(i,j)\in E : i,j\in S\},\ \text{for } S\subseteq N\\
&\delta(S) && \{(i,j)\in E : i\in S,\ j\notin S,\ \text{or}\ i\notin S,\ j\in S\}\\
&\delta(i) && \delta(\{i\}),\ \text{edges incident to node } i\\
&T=\{1,\ldots,l\} && \text{set of time periods}\\
&K=\{1,\ldots,m\} && \text{set of identical vehicles}
\end{align*}$$

## Parameters

$$\begin{align*}
&\rho_\omega && \text{probability of scenario } \omega\in\Omega\\
&d_{it\omega} && \text{demand of customer } i \text{ in period } t \text{ under scenario } \omega\\
&\sigma_i && \text{unit penalty cost for unmet demand at customer } i\\
&L_i && \text{maximum inventory level at customer } i\\
&L_0 && \text{maximum inventory level at the plant}\\
&I_{i0} && \text{initial inventory at node } i\ (I_{00\omega}=I_0\ \forall\omega)\\
&C && \text{production capacity per period}\\
&u && \text{unit production cost}\\
&f && \text{fixed production setup cost}\\
&Q && \text{vehicle capacity}\\
&c_{ij} && \text{transportation cost between nodes } i \text{ and } j\\
&h_i && \text{unit inventory holding cost at node } i\\
&M_{t\omega} && \min\Big\{C,\,Q,\,\textstyle\sum_{j=t}^{l}\sum_{i\in N_c} d_{ij\omega}\Big\}\\
&M'_{it\omega} && \min\Big\{L_i,\,Q,\,\textstyle\sum_{j=t}^{l} d_{ij\omega}\Big\}
\end{align*}$$

## Decision Variables

$$\begin{align*}
&y_t\in\{0,1\} && 1 \text{ if production takes place in period } t\\
&z_{ikt}\in\{0,1\} && 1 \text{ if node } i \text{ is visited by vehicle } k \text{ in period } t\\
&x_{ijkt} && \text{number of times vehicle } k \text{ travels directly between } i \text{ and } j \text{ in period } t\\
&p_{t\omega}\ge 0 && \text{production quantity in period } t \text{ under scenario } \omega\\
&I_{it\omega}\ge 0 && \text{inventory at node } i \text{ at end of period } t \text{ under scenario } \omega\\
&q_{ikt\omega}\ge 0 && \text{quantity delivered to customer } i \text{ by vehicle } k \text{ in period } t \text{ (scenario } \omega)\\
&e_{it\omega}\ge 0 && \text{unmet demand at customer } i \text{ in period } t \text{ under scenario } \omega
\end{align*}$$

## Objective

$$\begin{equation}
\min \sum_{t\in T}\left( f\,y_t + \sum_{(i,j)\in E}\sum_{k\in K} c_{ij}\,x_{ijkt}
  + \sum_{\omega\in\Omega} \rho_\omega\left( u\,p_{t\omega}
  + \sum_{i\in N} h_i\,I_{it\omega}
  + \sum_{i\in N_c} \sigma_i\,e_{it\omega} \right) \right)
\tag{1}
\end{equation}$$

## Constraints

$$\begin{align}
& I_{0,t-1,\omega} + p_{t\omega}
  = \sum_{i\in N_c}\sum_{k\in K} q_{ikt\omega} + I_{0t\omega}
  && \forall t\in T,\ \forall \omega\in\Omega \tag{2}\\[2pt]
& I_{i,t-1,\omega} + \sum_{k\in K} q_{ikt\omega} + e_{it\omega}
  = d_{it\omega} + I_{it\omega}
  && \forall i\in N_c,\ \forall t\in T,\ \forall \omega\in\Omega \tag{3}\\[2pt]
& I_{0t\omega} \le L_0
  && \forall t\in T,\ \forall \omega\in\Omega \tag{4}\\[2pt]
& I_{it\omega} + d_{it\omega} \le L_i
  && \forall i\in N_c,\ \forall t\in T,\ \forall \omega\in\Omega \tag{5}\\[2pt]
& p_{t\omega} \le M_{t\omega}\,y_t
  && \forall t\in T,\ \forall \omega\in\Omega \tag{6}\\[2pt]
& \sum_{i\in N_c} q_{ikt\omega} \le Q\,z_{0kt}
  && \forall k\in K,\ \forall t\in T,\ \forall \omega\in\Omega \tag{7}\\[2pt]
& q_{ikt\omega} \le M'_{it\omega}\,z_{ikt}
  && \forall i\in N_c,\ \forall k\in K,\ \forall t\in T,\ \forall \omega\in\Omega \tag{8}\\[2pt]
& \sum_{k\in K} z_{ikt} \le 1
  && \forall i\in N_c,\ \forall t\in T \tag{9}\\[2pt]
& \sum_{(j,j')\in\delta(i)} x_{jj'kt} = 2\,z_{ikt}
  && \forall i\in N,\ \forall k\in K,\ \forall t\in T \tag{10}\\[2pt]
& \sum_{(i,j)\in E(S)} x_{ijkt} \le \sum_{i\in S} z_{ikt} - z_{ekt}
  && \forall S\subseteq N_c:\ |S|\ge 2,\ \forall e\in S,\ \forall k\in K,\ \forall t\in T \tag{11}\\[2pt]
& e_{it\omega},\ p_{t\omega},\ I_{it\omega},\ q_{ikt\omega} \ge 0
  && \forall i\in N,\ \forall k\in K,\ \forall t\in T,\ \forall \omega\in\Omega \tag{12}\\[2pt]
& y_t,\ z_{ikt} \in \{0,1\}
  && \forall i\in N,\ \forall k\in K,\ \forall t\in T \tag{13}\\[2pt]
& x_{ijkt} \in \{0,1\}
  && \forall (i,j)\in E:\ i\neq 0,\ \forall k\in K,\ \forall t\in T \tag{14}\\[2pt]
& x_{0jkt} \in \{0,1,2\}
  && \forall j\in N_c,\ \forall k\in K,\ \forall t\in T \tag{15}
\end{align}$$

*Strengthening inequalities included by the paper inside the definition of Model (1)–(19) (2-BF): routing-strengthening (16)–(17) and vehicle symmetry-breaking (18)–(19).* $$\begin{align}
& z_{ikt} \le z_{0kt}
  && \forall i\in N_c,\ \forall k\in K,\ \forall t\in T \tag{16}\\[2pt]
& x_{ijkt} \le z_{ikt}\quad\text{and}\quad x_{ijkt} \le z_{jkt}
  && \forall (i,j)\in E(N_c),\ \forall k\in K,\ \forall t\in T \tag{17}\\[2pt]
& z_{0kt} \ge z_{0,k+1,t}
  && \forall\, 1\le k\le m-1,\ \forall t\in T \tag{18}\\[2pt]
& \sum_{i=1}^{j} 2^{(j-i)}\,z_{ikt} \ge \sum_{i=1}^{j} 2^{(j-i)}\,z_{i,k+1,t}
  && \forall j\in N_c,\ \forall\, 1\le k\le m-1,\ \forall t\in T \tag{19}
\end{align}$$

Model (1)–(19) is referred to as the **2-BF**.
