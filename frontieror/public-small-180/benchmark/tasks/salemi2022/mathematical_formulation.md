# Original Formulation: Unit Commitment Problem (UCP)

Salemi & Davarnia. “On the Structure of Decision Diagram-Representable Mixed Integer Programs with Application to Unit Commitment.” Formulation (1a)–(1l).

## Sets and Indices

$N = \{1,\dots,n\}$: set of generators, indexed by $i$.  
$\mathcal{T} = \{1,\dots,T\}$: set of time periods, indexed by $j$.

## Parameters

$c_f^i$: fixed operating (no-load) cost of generator $i$; $c_g^i$: variable production cost; $K_k^i$: start-up cost after $k$ consecutive inactive periods (linearization of a logarithmic start-up cost); $L^i \ge 1$: minimum up-time; $\ell^i \ge 1$: minimum down-time; $m^i$, $M^i$: minimum/maximum production capacity; $SU^i$, $SD^i$: start-up / shut-down ramp rates; $RU^i$, $RD^i$: ramp-up / ramp-down rates (assumed $SU^i\le RU^i$, $SD^i\le RD^i$); $D_j$: total demand at time $j$; $R_j$: spinning reserve requirement at time $j$.

## Decision Variables

$$\begin{align*}
  x_j^i          &\in \{0,1\}\quad
    \text{commitment: $=1$ iff unit $i$ is up at time $j$;} \\
  y_j^i          &\in \{0,1\}\quad
    \text{start-up: $=1$ iff unit $i$ starts up at time $j$;} \\
  \bar{y}_j^i    &\in \{0,1\}\quad
    \text{shut-down: $=1$ iff unit $i$ shuts down at time $j$;} \\
  p_j^i          &\ge 0\ \ \text{production of unit $i$ at time $j$;} \\
  \bar{p}_j^i    &\ge 0\ \ \text{maximum available power of unit $i$ at time $j$;} \\
  q_j^i          &\ge 0\ \ \text{start-up cost variable of unit $i$ at time $j$.}
\end{align*}$$

## Formulation (1)

$$\begin{align}
\min\quad
  & \sum_{i=1}^{n}\sum_{j=1}^{T} \bigl(c_f^i\, x_j^i + c_g^i\, p_j^i + q_j^i\bigr) \tag{1a}\\
\text{s.t.}\quad
  & q_j^i \ge K_k^i\!\left(x_j^i - \sum_{h=1}^{k} x_{j-h}^i\right),
    && \!\!\!\!\!\!\!\!\forall k\in\{1,\dots,j-1\},\, j\in\mathcal{T},\, i\in N \tag{1b}\\
  & y_j^i - \bar{y}_j^i = x_j^i - x_{j-1}^i,
    && \forall j\in\{1,\dots,T\},\, i\in N \tag{1c}\\
  & \sum_{j'=j-L^i+1}^{j} y_{j'}^i \le x_j^i,
    && \forall j\in\{L^i,\dots,T\},\, i\in N \tag{1d}\\
  & \sum_{j'=j-\ell^i+1}^{j} \bar{y}_{j'}^i \le 1 - x_j^i,
    && \forall j\in\{\ell^i,\dots,T\},\, i\in N \tag{1e}\\
  & p_j^i - p_{j-1}^i \le RU^i\, x_{j-1}^i + SU^i\, y_j^i,
    && \forall j\in\mathcal{T},\, i\in N \tag{1f}\\
  & p_{j-1}^i - p_j^i \le RD^i\, x_j^i + SD^i\, \bar{y}_j^i,
    && \forall j\in\mathcal{T},\, i\in N \tag{1g}\\
  & m^i\, x_j^i \le p_j^i \le \bar{p}_j^i \le M^i\, x_j^i,
    && \forall j\in\mathcal{T},\, i\in N \tag{1h}\\
  & \sum_{i=1}^{n} p_j^i \ge D_j,
    && \forall j\in\mathcal{T} \tag{1i}\\
  & \sum_{i=1}^{n} \bar{p}_j^i \ge D_j + R_j,
    && \forall j\in\mathcal{T} \tag{1j}\\
  & x_j^i,\ y_j^i,\ \bar{y}_j^i \in \{0,1\},
    && \forall j\in\mathcal{T},\, i\in N \tag{1k}\\
  & q_j^i,\ p_j^i,\ \bar{p}_j^i \ge 0,
    && \forall j\in\mathcal{T},\, i\in N. \tag{1l}
\end{align}$$
