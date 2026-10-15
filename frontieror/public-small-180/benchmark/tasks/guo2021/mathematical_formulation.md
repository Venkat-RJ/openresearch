# Original Formulation: Stochastic Distributed Operating Room Scheduling (SDORS)

*Source: “Logic-based Benders Decomposition and Binary Decision Diagram Based Approaches for Stochastic Distributed Operating Room Scheduling”, Cheng Guo, Merve Bodur, Dionne M. Aleman, David R. Urbach, 2021.*

## Sets and Indices

$$\begin{align*}
\mathcal{P}      &: \text{Set of patients, } p \in \mathcal{P}. \\
\mathcal{P}'     &: \text{Set of mandatory patients (those with health score } \omega_p \ge \Gamma\text{), } \mathcal{P}' \subseteq \mathcal{P}. \\
\mathcal{H}      &: \text{Set of hospitals, } h \in \mathcal{H}. \\
\mathcal{D}      &: \text{Set of days in the current planning horizon, } d \in \mathcal{D}. \\
\mathcal{R}_h    &: \text{Set of ORs in the surgical suite of hospital } h, \ r \in \mathcal{R}_h \text{ (homogeneous within } h\text{)}. \\
\mathcal{S}      &: \text{Set of possible scenarios of uncertain surgery durations, } s \in \mathcal{S}.
\end{align*}$$

## Parameters

$$\begin{align*}
G_{hd}              &: \text{Cost of opening the surgical suite of hospital } h \text{ on day } d. \\
F_{hd}              &: \text{Cost of opening an OR in hospital } h \text{ on day } d. \\
B_{hd}              &: \text{Operating time limit of each OR on day } d \text{ in hospital } h. \\
T_p                 &: \text{Total booked time of patient } p \text{ (component of the random duration vector } \mathbf{T}\text{)}. \\
c^{\text{sched}}_{dp}&: \text{Benefit for scheduling patient } p \text{ on day } d \ (\le 0, \text{ a negative cost under minimization)}. \\
c^{\text{unsched}}_p &: \text{Penalty for postponing (unscheduling) patient } p. \\
c^{\text{cancel}}_p  &: \text{Penalty for cancelling patient } p. \\
\omega_p             &: \text{Health status score of patient } p. \\
\Gamma               &: \text{Health score threshold separating mandatory from non-mandatory patients.}
\end{align*}$$

## Decision Variables

**First-stage (here-and-now) variables:** $$\begin{align*}
u_{hd}   &\in \{0,1\}: \ 1 \text{ if the surgical suite in hospital } h \text{ is opened on day } d,\ 0 \text{ otherwise.} \\
y_{hdr}  &\in \{0,1\}: \ 1 \text{ if OR } r \text{ of hospital } h \text{ is opened on day } d,\ 0 \text{ otherwise.} \\
x_{hdpr} &\in \{0,1\}: \ 1 \text{ if patient } p \text{ is assigned to OR } r \text{ of hospital } h \text{ on day } d,\ 0 \text{ otherwise.} \\
w_p      &\in \{0,1\}: \ 1 \text{ if patient } p \text{ is postponed to a future planning horizon,}\ 0 \text{ otherwise.}
\end{align*}$$ **Second-stage (recourse) variables:** $$\begin{align*}
z_{hdpr} &\in \{0,1\}: \ 1 \text{ if patient } p\text{'s surgery in OR } r \text{ of hospital } h \text{ on day } d \text{ is operated,}\ 0 \text{ if it is cancelled.}
\end{align*}$$

## Objective

The SDORS problem is the two-stage stochastic integer program (2SIP) whose first-stage objective minimizes the total operational plus expected cancellation cost: $$\begin{align}
(\text{SDORS}): \quad \min \quad
  & \sum_{h \in \mathcal{H}} \sum_{d \in \mathcal{D}} G_{hd}\, u_{hd}
  + \sum_{h \in \mathcal{H}} \sum_{d \in \mathcal{D}} \sum_{r \in \mathcal{R}_h} F_{hd}\, y_{hdr}
  + \sum_{h \in \mathcal{H}} \sum_{d \in \mathcal{D}} \sum_{p \in \mathcal{P}} \sum_{r \in \mathcal{R}_h} c^{\text{sched}}_{dp}\, x_{hdpr} \notag \\
  & + \sum_{p \in \mathcal{P} \setminus \mathcal{P}'} c^{\text{unsched}}_p\, w_p
  + \mathbb{E}_{\mathbf{T}}\, \mathcal{Q}(\mathbf{x}, \mathbf{y}, \mathbf{T})
  \tag{1a}
\end{align}$$

## Constraints

**First-stage constraints:** $$\begin{align}
\text{s.t.} \quad
  & \sum_{h \in \mathcal{H}} \sum_{d \in \mathcal{D}} \sum_{r \in \mathcal{R}_h} x_{hdpr} = 1
  && \forall p \in \mathcal{P}' \tag{1b} \\
  & \sum_{h \in \mathcal{H}} \sum_{d \in \mathcal{D}} \sum_{r \in \mathcal{R}_h} x_{hdpr} + w_p = 1
  && \forall p \in \mathcal{P} \setminus \mathcal{P}' \tag{1c} \\
  & y_{hdr} \le y_{hd,r-1}
  && \forall h \in \mathcal{H},\ d \in \mathcal{D},\ r \in \mathcal{R}_h \setminus \{1\} \tag{1d} \\
  & \sum_{p \in \mathcal{P}} c^{\text{cancel}}_p\, x_{hdpr}
    \le \sum_{p \in \mathcal{P}} c^{\text{cancel}}_p\, x_{hdp,r-1}
  && \forall h \in \mathcal{H},\ d \in \mathcal{D},\ r \in \mathcal{R}_h \setminus \{1\} \tag{1e} \\
  & y_{hdr} \le u_{hd}
  && \forall h \in \mathcal{H},\ d \in \mathcal{D},\ r \in \mathcal{R}_h \tag{1f} \\
  & x_{hdpr} \le y_{hdr}
  && \forall h \in \mathcal{H},\ d \in \mathcal{D},\ p \in \mathcal{P},\ r \in \mathcal{R}_h \tag{1g} \\
  & u_{hd},\ y_{hdr},\ x_{hdpr},\ w_p \in \{0, 1\}
  && \forall h \in \mathcal{H},\ d \in \mathcal{D},\ p \in \mathcal{P},\ r \in \mathcal{R}_h \tag{1h}
\end{align}$$

**Second-stage recourse problem** $\mathcal{Q}(\mathbf{x}, \mathbf{y}, \mathbf{T})$, parameterized by the first-stage decisions $\mathbf{x} = \{x_{hdpr}\}$, $\mathbf{y} = \{y_{hdr}\}$ and the realized duration vector $\mathbf{T} = \{T_1, \dots, T_{|\mathcal{P}|}\}$: $$\begin{align}
\mathcal{Q}(\mathbf{x}, \mathbf{y}, \mathbf{T}) = \min \quad
  & \sum_{h \in \mathcal{H}} \sum_{d \in \mathcal{D}} \sum_{p \in \mathcal{P}} \sum_{r \in \mathcal{R}_h}
    c^{\text{cancel}}_p\, (x_{hdpr} - z_{hdpr})
  \tag{2a} \\
\text{s.t.} \quad
  & \sum_{p \in \mathcal{P}} T_p\, z_{hdpr} \le B_{hd}\, y_{hdr}
  && \forall h \in \mathcal{H},\ d \in \mathcal{D},\ r \in \mathcal{R}_h \tag{2b} \\
  & z_{hdpr} \le x_{hdpr}
  && \forall h \in \mathcal{H},\ d \in \mathcal{D},\ p \in \mathcal{P},\ r \in \mathcal{R}_h \tag{2c} \\
  & z_{hdpr} \in \{0, 1\}
  && \forall h \in \mathcal{H},\ d \in \mathcal{D},\ p \in \mathcal{P},\ r \in \mathcal{R}_h \tag{2d}
\end{align}$$
