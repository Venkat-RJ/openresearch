# Original Formulation: Vertex Coloring Problem (VCP)

*Source: An Improved DSATUR-Based Branch-and-Bound Algorithm for the Vertex Coloring Problem, Fabio Furini, Virginie Gabrel, Ian-Christopher Ternier, 2017.*

## Sets and Parameters

$$\begin{align*}
G=(V,E) &: \text{undirected graph the problem is defined on} \\
V       &: \text{set of vertices, } |V|=n \\
E       &: \text{set of edges; } (u,v)\in E \text{ iff } u,v \text{ are adjacent} \\
N(u)    &: \text{neighborhood of } u,\ N(u)=\{v\in V:(u,v)\in E\} \\
H       &: \text{set of candidate color indices, } H=\{1,\dots,\bar{k}\}
           \text{ with } \bar{k}\le n
           \quad\text{\% RECONSTRUCTED --- not in paper}
\end{align*}$$

## Decision Variables

$$\begin{align*}
x_{v,i} &\in\{0,1\}, \quad v\in V,\ i\in H, \quad
   x_{v,i}=1 \text{ iff vertex } v \text{ receives color } i
   && \text{\% RECONSTRUCTED --- not in paper} \\
w_{i}   &\in\{0,1\}, \quad i\in H, \quad
   w_{i}=1 \text{ iff color } i \text{ is used by some vertex}
   && \text{\% RECONSTRUCTED --- not in paper}
\end{align*}$$

## Objective

$$\begin{equation}
\min \ \sum_{i\in H} w_{i}
\tag{R1} % RECONSTRUCTED --- not in paper
\end{equation}$$

## Constraints

$$\begin{align}
\sum_{i\in H} x_{v,i} &= 1
   && \forall\, v\in V
   \tag{R2} \\ % RECONSTRUCTED: every vertex gets exactly one color
x_{u,i} + x_{v,i} &\le w_{i}
   && \forall\, (u,v)\in E,\ \forall\, i\in H
   \tag{R3} \\ % RECONSTRUCTED: adjacent vertices differ; links x to w
x_{v,i} &\in \{0,1\}
   && \forall\, v\in V,\ \forall\, i\in H
   \tag{R4} \\ % RECONSTRUCTED
w_{i} &\in \{0,1\}
   && \forall\, i\in H
   \tag{R5} % RECONSTRUCTED
\end{align}$$
