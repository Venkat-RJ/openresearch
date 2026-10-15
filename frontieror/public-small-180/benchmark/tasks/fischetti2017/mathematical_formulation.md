# Sets and Parameters

- $I$: set of potential facility locations, $|I|=n$.

- $J$: set of customers, $|J|=m$.

- $f_i \geq 0$: opening cost for facility $i \in I$.

- $c_{ij} \geq 0$: allocation cost (linear case) / per-unit quadratic cost (quadratic case) for assigning customer $j \in J$ to facility $i \in I$. If customer $j$ cannot be allocated to facility $i$ we set $c_{ij}=\infty$.

# 2.1 Linear Case (Classical UFL)

## Decision Variables

$$\begin{align*}
y_i &\in \{0,1\} && \forall\, i \in I \quad (\text{1 if facility $i$ is open}) \\
x_{ij} &\geq 0 && \forall\, i \in I,\; j \in J \quad (\text{fraction / assignment of customer $j$ to facility $i$})
\end{align*}$$

## Objective

$$\begin{equation}
\min \left\{ \sum_{i \in I} f_i \, y_i + \sum_{i \in I}\sum_{j \in J} c_{ij}\, x_{ij} \right\} \tag{1}
\end{equation}$$

## Constraints

$$\begin{align}
\sum_{i \in I} x_{ij} &= 1 && \forall\, j \in J \tag{2} \\
x_{ij} &\leq y_i && \forall\, i \in I,\; j \in J \tag{3} \\
x_{ij} &\geq 0 && \forall\, i \in I,\; j \in J \notag \\
y_i &\in \{0,1\} && \forall\, i \in I \notag
\end{align}$$

# 2.2 Separable Convex Quadratic Case (qUFL)

In the quadratic version $c_{ij} > 0$ for all $i \in I$, $j \in J$, and allocation costs are proportional to the square of the fraction of demand served. The objective $(1)$ is replaced by $(4)$.

## Decision Variables

$$\begin{align*}
y_i &\in \{0,1\} && \forall\, i \in I \\
x_{ij} &\geq 0 && \forall\, i \in I,\; j \in J \quad (\text{fraction of customer $j$'s demand served by facility $i$})
\end{align*}$$

## Objective (original quadratic form)

$$\begin{equation}
\min \left\{ \sum_{i \in I} f_i\, y_i + \sum_{i \in I}\sum_{j \in J} c_{ij}\, x_{ij}^{2} \right\} \tag{4}
\end{equation}$$

## Constraints

Same as constraints (2)–(3) together with $x_{ij} \geq 0$ and $y_i \in \{0,1\}$: $$\begin{align*}
\sum_{i \in I} x_{ij} &= 1 && \forall\, j \in J \\
x_{ij} &\leq y_i && \forall\, i \in I,\; j \in J \\
x_{ij} &\geq 0 && \forall\, i \in I,\; j \in J \\
y_i &\in \{0,1\} && \forall\, i \in I
\end{align*}$$
