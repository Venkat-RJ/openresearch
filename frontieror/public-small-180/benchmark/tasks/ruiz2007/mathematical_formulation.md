# Original Formulation: Mixed No-Idle Permutation Flowshop Scheduling Problem (MNPFSP)

*Source: An effective iterated greedy algorithm for the mixed no-idle permutation flowshop scheduling problem, Quan-Ke Pan and Rubén Ruiz, 2014 (Omega 44:41–50; preprint submitted to OMEGA, October 2013).*

## Sets and Parameters

$$\begin{align*}
N &= \{1,\ldots,n\} && \text{set of } n \text{ jobs; } j \in N \text{ is a job index}\\
M &= \{1,\ldots,m\} && \text{set of } m \text{ machines; } i \in M \text{ is a machine index}\\
M' &\subseteq M && \text{subset of no-idle machines, } |M'| = m'\\
&&& \text{(machines not in } M' \text{ are regular idle machines)}\\
k &\in \{1,\ldots,n\} && \text{position index in the permutation sequence}\\
p_{i,j} &\geq 0 && \text{known, deterministic processing time of job } j \text{ on machine } i
\end{align*}$$

## Decision Variables

$$\begin{align*}
X_{j,k} &=
\begin{cases}
1, & \text{if job } j \text{ occupies position } k \text{ of the sequence}\\
0, & \text{otherwise}
\end{cases}\\[4pt]
C_{i,k} &= \text{completion time of the job in position } k \text{ on machine } i\\
C_{\max} &= \text{maximum completion time (makespan)}
\end{align*}$$

## Objective

The objective is the minimization of the makespan, equal to the time at which the job occupying the last position of the permutation finishes at the last machine: $$\begin{equation*}
\min \; C_{\max} = C_{m,n}
\end{equation*}$$

## Constraints

Subject to the following constraints: $$\begin{align}
&\sum_{k=1}^{n} X_{j,k} = 1, && j = 1,\ldots,n \tag{1}\\[4pt]
&\sum_{j=1}^{n} X_{j,k} = 1, && k = 1,\ldots,n \tag{2}\\[4pt]
&C_{1,k} \geq \sum_{j=1}^{n} X_{j,1}\cdot p_{1,j}, && k = 1,\ldots,n \tag{3}\\[4pt]
&C_{i,k} \geq C_{i-1,k} + \sum_{j=1}^{n} X_{j,k}\cdot p_{i,j}, && k = 1,\ldots,n,\; i = 2,\ldots,m \tag{4}\\[4pt]
&\begin{cases}
C_{i,k} = C_{i,k-1} + \displaystyle\sum_{j=1}^{n} X_{j,k}\cdot p_{i,j}, & \text{if } i \in M'\\[10pt]
C_{i,k} \geq C_{i,k-1} + \displaystyle\sum_{j=1}^{n} X_{j,k}\cdot p_{i,j}, & \text{otherwise}
\end{cases}
&& k = 2,\ldots,n,\; i = 1,\ldots,m \tag{5}\\[4pt]
&C_{i,k} \geq 0, && k = 1,\ldots,n,\; i = 1,\ldots,m \tag{6}\\[4pt]
&X_{i,k} \in \{0,1\}, && k = 1,\ldots,n,\; i = 1,\ldots,m \tag{7}
\end{align}$$
