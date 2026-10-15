# Sets and Parameters

- $m$: number of machines; index $i = 1,\ldots,m$.

- $n$: number of jobs; index $j,k = 1,\ldots,n$. Job $j$ is denoted $J_j$.

- $p_{ij}$: processing time of the $i$-th operation of $J_j$.

- $d_j$: due date of $J_j$.

- $M$: a sufficiently large constant (big-$M$).

# Decision Variables

$$\begin{equation}
x_{jk} = \begin{cases} 1 & \text{if } J_k \text{ is placed immediately after } J_j \text{ in the sequence}, \\ 0 & \text{otherwise}, \end{cases}
\quad j,k = 1,2,\ldots,n \tag{3}
\end{equation}$$ Continuous variables: $S_{ij} \geq 0$ (starting time of the $i$-th operation of $J_j$); $C_{\max} \geq 0$.

# Objective

$$\begin{equation}
\text{minimize} \quad C_{\max} \tag{4}
\end{equation}$$

# Constraints

$$\begin{align}
C_{\max} &\geq S_{mj} + p_{mj} && j = 1,2,\ldots,n \tag{5} \\
S_{ik} + M\bigl(1 - x_{jk}\bigr) &\geq S_{ij} + p_{ij} && i = 1,\ldots,m;\; j,k = 1,\ldots,n \tag{6} \\
S_{(i+1)j} &= S_{ij} + p_{ij} && i = 1,\ldots,m-1;\; j = 1,\ldots,n \tag{7} \\
S_{mj} + p_{mj} &\leq d_j && j = 1,2,\ldots,n \tag{8} \\
\sum_{j=1}^{n} x_{jk} &\leq 1 && k = 1,2,\ldots,n \tag{9} \\
\sum_{k=1}^{n} x_{jk} &\leq 1 && j = 1,2,\ldots,n \tag{10} \\
x_{jk} + x_{kj} &\leq 1 && j,k = 1,2,\ldots,n \tag{11} \\
\sum_{j=1}^{n}\sum_{k=1}^{n} x_{jk} &= n-1 \tag{12} \\
x_{jj} &= 0 && j = 1,2,\ldots,n \tag{13} \\
S_{ij} &\geq 0 && i = 1,\ldots,m;\; j = 1,\ldots,n \tag{14} \\
x_{jk} &\in \{0,1\} && j,k = 1,2,\ldots,n \tag{15}
\end{align}$$
