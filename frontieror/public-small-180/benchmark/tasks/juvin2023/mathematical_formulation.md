# Sets and Indices

- $\mathcal{J}$: set of jobs.

- $\mathcal{M}$: set of machines.

- $n_i$: number of operations of job $i \in \mathcal{J}$.

- $\mathcal{O}_i$: set of operations of job $i$; $O_{i,j}$ is the $j$-th operation.

- $\mathcal{M}_{i,j}$: eligible machines for operation $O_{i,j}$.

- $\mathcal{I}_m$: operations processable on machine $m \in \mathcal{M}$.

- $\mathcal{H} = \{1,2,\ldots,h\}$: discrete time horizon.

# Parameters

- $p_{i,j,m}$: processing time of $O_{i,j}$ on machine $m \in \mathcal{M}_{i,j}$.

# Decision Variables

- $x_{i,j,m} \in \{0,1\}$: $=1$ iff $O_{i,j}$ is assigned to machine $m$.

- $y_{i,j,t} \in \{0,1\}$: $=1$ iff $O_{i,j}$ is in process at time $t$.

- $z_{i,j,m,t} \in \{0,1\}$: linearisation variable, $=1$ iff $O_{i,j}$ is processed on machine $m$ at time $t$.

- $C_{\max} \geq 0$: makespan.

# Objective

$$\begin{align}
\min \quad & C_{\max}. \tag{1}
\end{align}$$

# Original Constraints (with Bilinear (5))

$$\begin{align}
& \sum_{m \in \mathcal{M}_{i,j}} x_{i,j,m} = 1,
   && \forall i \in \mathcal{J},\ O_{i,j} \in \mathcal{O}_i, \tag{2}\\
& \sum_{t=1}^{h} y_{i,j,t} \geq \sum_{m \in \mathcal{M}_{i,j}} x_{i,j,m}\, p_{i,j,m},
   && \forall i \in \mathcal{J},\ O_{i,j} \in \mathcal{O}_i, \tag{3}\\
& \sum_{t'=t}^{h} y_{i,j,t'} \leq \max_{m \in \mathcal{M}_{i,j}} p_{i,j,m}\;\bigl(1 - y_{i,j+1,t}\bigr),
   && \forall i \in \mathcal{J},\ O_{i,j} \in \mathcal{O}_i\setminus\{O_{i,n_i}\},\ t \in \mathcal{H}, \tag{4}\\
& \sum_{i \in \mathcal{J}} \sum_{j=1}^{n_i} x_{i,j,m}\, y_{i,j,t} \leq 1,
   && \forall m \in \mathcal{M},\ t \in \mathcal{H}, \tag{5}\\
& C_{\max} \geq (t+1)\sum_{i \in \mathcal{J}} y_{i,n_i,t},
   && \forall t \in \mathcal{H}, \tag{6}\\
& x_{i,j,m} \in \{0,1\},
   && \forall i \in \mathcal{J},\ O_{i,j} \in \mathcal{O}_i,\ m \in \mathcal{M}_{i,j}, \tag{7}\\
& y_{i,j,t} \in \{0,1\},
   && \forall i \in \mathcal{J},\ O_{i,j} \in \mathcal{O}_i,\ t \in \mathcal{H}. \tag{8}
\end{align}$$

Constraint (5) contains the product $x_{i,j,m}\,y_{i,j,t}$ of two binary variables and is therefore bilinear. The paper linearises (5) in the same Section 2.1 via the auxiliary binary $z_{i,j,m,t}$ as follows.

# Paper’s Own Linearisation of Constraint (5)

$$\begin{align}
& z_{i,j,m,t} \geq x_{i,j,m} + y_{i,j,t} - 1,
   && \forall i \in \mathcal{J},\ O_{i,j} \in \mathcal{O}_i,\ m \in \mathcal{M}_{i,j},\ t \in \mathcal{H}, \tag{9}\\
& z_{i,j,m,t} \leq x_{i,j,m},
   && \forall i \in \mathcal{J},\ O_{i,j} \in \mathcal{O}_i,\ m \in \mathcal{M}_{i,j},\ t \in \mathcal{H}, \tag{10}\\
& z_{i,j,m,t} \leq y_{i,j,t},
   && \forall i \in \mathcal{J},\ O_{i,j} \in \mathcal{O}_i,\ m \in \mathcal{M}_{i,j},\ t \in \mathcal{H}, \tag{11}\\
& \sum_{i \in \mathcal{J}} \sum_{j=1}^{n_i} z_{i,j,m,t} \leq 1,
   && \forall m \in \mathcal{M},\ t \in \mathcal{H}. \tag{12}
\end{align}$$

The linearised formulation replaces the bilinear (5) with (9)–(12), yielding an equivalent MILP.
