# Original Formulation: Demand-oriented Rail Rapid Transit Timetabling Problem (MINLP)

*Source: Variable and adaptive neighbourhood search algorithms for rail rapid transit timetabling problem, Erfan Hassannayebi, Seyed Hessameddin Zegordi, 2016 (Computers & Operations Research).*

The paper first defines the studied problem as a capacitated train timetabling model under dynamic (multi-period) demand, written as a mixed-integer nonlinear programming (MINLP) model with continuous departure times, equations (1)–(18). This is the canonical formulation whose notation ($d_i, H_i, \delta_i, \Delta_i, w_i, b_i, y_i^{(p)}$) is carried into all later sections (the time-indexed MILP, the mini-max variant, and the MINLP$_d$ decoding model used inside the neighbourhood-search heuristic are downstream alternatives, collected under *Variants*).

## Sets and Indices

$$\begin{align*}
I    &: \text{set of train services, indexed by } i \;(i = 0,1,\dots,D);\quad |I| = D \text{ services} \\
P    &: \text{set of demand periods, indexed by } p \;(\text{or } p',\,t,\,t');\quad NT = |P| \text{ periods} \\
[0,T] &: \text{planning horizon (period of service)}
\end{align*}$$

## Parameters

$$\begin{align*}
\lambda^{(p)} &: \text{passenger arrival rate (passengers/minute) during period } p \in P \\
\theta_p      &: \text{length of demand period } p \text{ (minutes)} \\
t_p           &: \text{start time of demand period } p \\
h_{\min}      &: \text{minimum allowed headway (minutes)} \\
h_{\max}      &: \text{maximum allowed headway (minutes)} \\
C             &: \text{train capacity (maximum on-board passengers)} \\
T             &: \text{total length of the planning horizon (minutes)} \\
M             &: \text{sufficiently large constant (Big-}M\text{; value not specified in paper)}
\end{align*}$$

## Decision Variables

$$\begin{align*}
d_i        &: \text{departure time of the } i\text{-th train service} \quad (\text{continuous}, \ge 0) \\
H_i        &: \text{headway between the } i\text{-th and } (i{+}1)\text{-th departures} \quad (\text{continuous}, \ge 0) \\
\delta_i   &: \text{number of passengers arriving between departures } i \text{ and } i{+}1 \quad (\text{continuous}, \ge 0) \\
\Delta_i   &: \text{cumulative passengers arrived up to the departure of train } i \quad (\text{continuous}, \ge 0) \\
w_i        &: \text{number of waiting passengers after departure of train } i \quad (\text{continuous}, \ge 0) \\
b_i        &: \text{number of on-board (boarding) passengers on train } i \quad (\text{continuous}, \ge 0) \\
y_i^{(p)}  &: 1 \text{ if the } i\text{-th train runs at demand period } p, \ 0 \text{ otherwise} \quad (\text{binary})
\end{align*}$$

## Objective

$$\begin{align}
\textbf{[MINLP]:}\quad \min \; z &= \sum_{i \in I} \left(w_i + \tfrac{1}{2}\delta_i\right) \cdot H_i \tag{1}
\end{align}$$ *Bilinear (nonlinear) objective: product of the waiting/arrival term and the headway $H_i$.*

## Constraints

$$\begin{align}
\sum_{i \in I} H_i &= T \tag{2} \\[2pt]
H_i &= d_{i+1} - d_i, \quad i \in I \setminus \{D\} \tag{3} \\[2pt]
h_{\min} \;\le\; H_i &\le\; h_{\max}, \quad i \in I \setminus \{0, D\} \tag{4} \\[2pt]
t_{p-1} - M\,\bigl(1 - y_i^{(p)}\bigr) \;\le\; d_i &\;<\; t_p + M\,\bigl(1 - y_i^{(p)}\bigr), \quad i \in I,\; p \in P \tag{5} \\[2pt]
\sum_{p \in P} y_i^{(p)} &= 1, \quad i \in I \tag{6} \\[2pt]
y_{i+1}^{(p')} + y_i^{(p)} &\le 1, \quad p = 2,3,\ldots,|P|,\; p' \le p-1,\; i \in I \tag{7} \\[2pt]
\delta_{i+1} &= \Delta_{i+1} - \Delta_i, \quad i \in I \setminus \{0, D\} \tag{8} \\[2pt]
w_i &= w_{i-1} + \delta_{i-1} - b_i, \quad i \in I \tag{9} \\[2pt]
b_i &\le C, \quad i \in I \tag{10}
\end{align}$$

Cumulative arrivals $\Delta_i$ up to the $i$-th departure, $i \in I \setminus \{0\}$ (equality enforced by the two-sided bound): $$\begin{align}
\Delta_i &\le \sum_{t' \in P} \Bigl(1 - \sum_{t=1}^{t'} y_i^{(t)}\Bigr)\,\theta_{t'}\,\lambda^{(t')}
              + \bigl(d_i - t_{p-1}\bigr)\,\lambda^{(p)} + M\,\bigl(1 - y_i^{(p)}\bigr),
              \quad i \in I \setminus \{0\},\; p \in P \tag{11} \\[2pt]
\Delta_i &\ge \sum_{t' \in P} \Bigl(1 - \sum_{t=1}^{t'} y_i^{(t)}\Bigr)\,\theta_{t'}\,\lambda^{(t')}
              + \bigl(d_i - t_{p-1}\bigr)\,\lambda^{(p)} + M\,\bigl(1 - y_i^{(p)}\bigr),
              \quad i \in I \setminus \{0\},\; p \in P \tag{12}
\end{align}$$

Special case — passengers arriving before the first train service ($\delta_0$): $$\begin{align}
\delta_0 &\le \sum_{t' \in P} \Bigl(1 - \sum_{t=1}^{t'} y_0^{(t)}\Bigr)\,\theta_{t'}\,\lambda^{(t')}
              + \bigl(d_1 - t_{p-1}\bigr)\,\lambda^{(p)} + M\,\bigl(1 - y_0^{(p)}\bigr), \quad p \in P \tag{13} \\[2pt]
\delta_0 &\ge \sum_{t' \in P} \Bigl(1 - \sum_{t=1}^{t'} y_0^{(t)}\Bigr)\,\theta_{t'}\,\lambda^{(t')}
              + \bigl(d_1 - t_{p-1}\bigr)\,\lambda^{(p)} + M\,\bigl(1 - y_0^{(p)}\bigr), \quad p \in P \tag{14}
\end{align}$$

Special case — passengers arriving after the last train service $D$ ($\delta_D$): $$\begin{align}
\delta_D &\le \sum_{t' \in P} \Bigl(1 - \sum_{t=t'}^{NT} y_D^{(t)}\Bigr)\,\theta_{t'}\,\lambda^{(t')}
              + \bigl(t_p - d_D\bigr)\,\lambda^{(p)} + M\,\bigl(1 - y_D^{(p)}\bigr), \quad p \in P \tag{15} \\[2pt]
\delta_D &\ge \sum_{t' \in P} \Bigl(1 - \sum_{t=t'}^{NT} y_D^{(t)}\Bigr)\,\theta_{t'}\,\lambda^{(t')}
              + \bigl(t_p - d_D\bigr)\,\lambda^{(p)} + M\,\bigl(1 - y_D^{(p)}\bigr), \quad p \in P \tag{16}
\end{align}$$

Initial conditions and variable domains: $$\begin{align}
w_0 &= b_0 = 0 \tag{17} \\[2pt]
d_i, \delta_i, \Delta_i, w_i, b_i, H_i &\in \mathbb{R}^{+} \;\; \forall i \in I,
   \qquad y_i^{(p)} \in \{0,1\} \;\; \forall i \in I,\; p \in P \tag{18}
\end{align}$$
