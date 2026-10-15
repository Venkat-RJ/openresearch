# Original Formulation: Attractive Orienteering Problem (AtOP)

*Source: The Attractive Orienteering Problem with Proximity and Timing Interactions, M. C. Freeman, B. B. Keskin, İ. Çapar, European Journal of Operational Research, 2018.*

## Sets and Indices

$$\begin{align*}
& i \in \mathcal{I} = \{1,\ldots,I\} && \text{customer location index} \\
& j,k \in \mathcal{K} = \{1,\ldots,K\} && \text{site location indices} \\
& t = 1,\ldots,T && \text{period index}
\end{align*}$$ Site $0$ is a dummy starting site with $d^s_{0k}=0$ for all $k\in\mathcal{K}$.

## Parameters

$$\begin{align*}
& d^s_{jk}     && \text{distance between site } j \text{ and site } k, && \forall j,k\in\mathcal{K} \\
& d^p_{ik}     && \text{distance between customer } i \text{ and site } k, && \forall i\in\mathcal{I},\,k\in\mathcal{K} \\
& c            && \text{cost per unit distance traveled} \\
& \delta       && \text{maximum consecutive events} \\
& m_k          && \text{maximum number of events that may be held at site } k, && \forall k\in\mathcal{K} \\
& r_k          && \text{revenue per customer that attends an event at site } k, && \forall k\in\mathcal{K} \\
& f_k          && \text{fixed cost of holding an event at site } k, && \forall k\in\mathcal{K} \\
& \gamma_i     && \text{customer population at location } i, && \forall i\in\mathcal{I} \\
& L            && \text{maximum travel distance for a single period} \\
& \theta_k     && \text{capacity of site } k, && \forall k\in\mathcal{K} \\
& s_{it}       && \text{self-attraction of customer } i \text{ in period } t, && \forall i\in\mathcal{I},\,t=1,\ldots,T \\
& \alpha_{ikt} && \text{attraction of customers at location } i \text{ to an} \\
&              && \text{event held at site } k \text{ in period } t, && \forall i\in\mathcal{I},\,k\in\mathcal{K},\,t=1,\ldots,T \\
& \epsilon     && \text{number of periods before/after } t \text{ defining the} \\
&              && \text{performance shadow window}
\end{align*}$$ The attraction parameter is defined (Eq. 1 context) as $\alpha_{ikt} = \dfrac{\tilde{t}\,a_{kt}}{(d^p_{ik})^q}$, where $\tilde{t}\in[0,1]$ is the relative attraction of period $t$, $a_{kt}$ is the attraction of site $k$ in period $t$, and $q>1$ is the distance decay exponent.

## Decision Variables

$$\begin{align*}
& X_{jkt} \in \{0,1\} && \text{1 if the tour moves from site } j \text{ to site } k \text{ in period } t, && \forall j,k\in\mathcal{K},\,t=1,\ldots,T \\
& Y_{jt}  \in \{0,1\} && \text{1 if an event is held at site } j \text{ in period } t, && \forall j\in\mathcal{K},\,t=1,\ldots,T
\end{align*}$$ Auxiliary variables: $$\begin{align*}
& P_{ikt} && \text{expected proportion of the population at customer } i\text{'s location} \\
&         && \text{that will attend an event held at venue } k \text{ on day } t, && \forall i\in\mathcal{I},\,k\in\mathcal{K},\,t=1,\ldots,T \\
& A_{kt}  && \text{expected total attendance of an event held at facility } k \text{ on day } t, && \forall k\in\mathcal{K},\,t=1,\ldots,T
\end{align*}$$

## Objective

$$\begin{align}
\text{Maximize} \quad
\sum_{k\in\mathcal{K}}\sum_{t=1}^{T} r_k A_{kt}
- \sum_{k\in\mathcal{K}}\sum_{t=1}^{T} f_k Y_{kt}
- c \sum_{j\in\mathcal{K}}\sum_{k\in\mathcal{K}}\sum_{t=1}^{T} d^s_{jk} X_{jkt}
\notag
\end{align}$$

## Constraints

$$\begin{align}
& \sum_{k\in\mathcal{K}} X_{0k1} = 1 \tag{2} \\
& \sum_{j\in\mathcal{K}}\sum_{k\in\mathcal{K}} X_{jkt} = 1, && \forall t = 2,\ldots,T \tag{3} \\
& X_{0k1} = \sum_{j\in\mathcal{K}} X_{kj2}, && \forall k\in\mathcal{K} \tag{4} \\
& \sum_{j\in\mathcal{K}} X_{jkt} = \sum_{j\in\mathcal{K}} X_{kj,t+1}, && \forall k\in\mathcal{K},\,t=2,\ldots,T-1 \tag{5} \\
& \sum_{j\in\mathcal{K}}\sum_{k\in\mathcal{K}} X_{jkt}\, d^s_{jk} \leq L, && \forall t = 1,\ldots,T \tag{6} \\
& Y_{k1} \leq X_{0k1}, && \forall k\in\mathcal{K} \tag{7} \\
& Y_{kt} \leq \sum_{j\in\mathcal{K}} X_{jkt}, && \forall k\in\mathcal{K},\,t=2,\ldots,T \tag{8} \\
& \sum_{t=1}^{T} Y_{kt} \leq m_k, && \forall k\in\mathcal{K} \tag{9} \\
& \sum_{k\in\mathcal{K}}\sum_{t'=t}^{t+\delta+1} Y_{kt'} \leq \delta, && \forall t = 1,\ldots,T-\delta-1 \tag{10}
\end{align}$$

$$\begin{align}
& \left( s_{it} + \sum_{j\in\mathcal{K}} \sum_{\tau=\max\{t-\epsilon,0\}}^{\min\{t+\epsilon,T\}} \alpha_{ij\tau} Y_{j\tau} \right) P_{ikt} = \alpha_{ikt} Y_{kt}, && \forall i\in\mathcal{I},\,k\in\mathcal{K},\,t=1,\ldots,T \tag{11}
\end{align}$$

$$\begin{align}
& A_{kt} \leq \sum_{i\in\mathcal{I}} P_{ikt} \gamma_i, && \forall k\in\mathcal{K},\,t=1,\ldots,T \tag{12} \\
& A_{kt} \leq \theta_k, && \forall k\in\mathcal{K},\,t=1,\ldots,T \tag{13} \\
& X_{jkt} \in \{0,1\}, && \forall j,k\in\mathcal{K},\,t=1,\ldots,T \tag{14} \\
& Y_{kt} \in \{0,1\}, && \forall k\in\mathcal{K},\,t=1,\ldots,T \tag{15} \\
& P_{ikt} \leq 1, && \forall i\in\mathcal{I},\,k\in\mathcal{K},\,t=1,\ldots,T \tag{16} \\
& A_{kt} \geq 0, && \forall k\in\mathcal{K},\,t=1,\ldots,T \tag{17}
\end{align}$$

## Linearization (Section 3.2, paper’s own)

The nonlinear constraint (11) is replaced by (18)–(22) via a new continuous variable $Z_{ijkt\tau} \geq 0$ that represents the product $P_{ikt}\,\alpha_{ij\tau} Y_{j\tau}$, with $\tau \in \Omega = \{\max\{0,t-\epsilon\},\ldots,\min\{t+\epsilon,T\}\}$. $$\begin{align}
& P_{ikt}\, s_{it} = \alpha_{ikt} Y_{kt} - \sum_{j\in\mathcal{K}} \sum_{\tau=\max\{t-\epsilon,0\}}^{\min\{t+\epsilon,T\}} Z_{ijkt\tau}, && \forall i\in\mathcal{I},\,k\in\mathcal{K},\,t=1,\ldots,T \tag{18} \\
& Z_{ijkt\tau} \leq P_{ikt}\, \alpha_{ij\tau}, && \forall i\in\mathcal{I},\,j,k\in\mathcal{K},\,t=1,\ldots,T,\,\tau\in\Omega \tag{19} \\
& Z_{ijkt\tau} \leq \alpha_{ij\tau} Y_{j\tau}, && \forall i\in\mathcal{I},\,j,k\in\mathcal{K},\,t=1,\ldots,T,\,\tau\in\Omega \tag{20} \\
& Z_{ijkt\tau} \geq P_{ikt}\, \alpha_{ij\tau} - \alpha_{ij\tau}\,(1 - Y_{j\tau}), && \forall i\in\mathcal{I},\,j,k\in\mathcal{K},\,t=1,\ldots,T,\,\tau\in\Omega \tag{21} \\
& Z_{ijkt\tau} \geq 0, && \forall i\in\mathcal{I},\,j,k\in\mathcal{K},\,t=1,\ldots,T,\,\tau\in\Omega \tag{22}
\end{align}$$
