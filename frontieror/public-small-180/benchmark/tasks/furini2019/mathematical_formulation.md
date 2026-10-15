# Sets and Parameters

- $G = (V, E)$: undirected graph with vertex set $V = \{v_1, v_2, \ldots, v_n\}$, $|V| = n$, and edge set $E$.

- $\bar{G} = (V, \bar{E})$: complement graph of $G$, with $\bar{E} = \{(v_i, v_j) \colon v_i, v_j \in V,\; i \neq j,\; (v_i, v_j) \notin E\}$.

# Decision Variables

$$\begin{align*}
x_i &\in \{0,1\} && i = 1,\ldots,n \quad (\text{1 if vertex $v_i$ is in the clique, 0 otherwise})
\end{align*}$$

# Objective

$$\begin{equation}
\text{minimize} \quad f(\mathbf{x}) \;=\; -\sum_{i=1}^{n} x_i \tag{3}
\end{equation}$$

# Constraints

$$\begin{align}
x_i + x_j &\leq 1 && \forall\, (v_i, v_j) \in \bar{E} \tag{3a} \\
x_i &\in \{0,1\} && i = 1,\ldots,n \tag{3b}
\end{align}$$

A solution $\mathbf{x}^{*}$ defines a maximum clique $C$ by: $v_i \in C$ iff $x_i^{*} = 1$; the cardinality of $C$ equals $-f(\mathbf{x}^{*})$. The number of constraints equals $|\bar{E}| = n(n-1)/2 - |E|$ (Equation (4)).
