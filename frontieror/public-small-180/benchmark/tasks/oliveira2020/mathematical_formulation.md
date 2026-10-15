# Original Formulation: Arc-Time-Indexed Formulation (ATIF)

**Source.** Oliveira and Pessoa (2020), “An Improved Branch-Cut-and-Price Algorithm for Parallel Machine Scheduling Problems,” *INFORMS Journal on Computing*. Section 2.1, equations (1a)–(1g).

**Problem.** Identical parallel machine scheduling $P\,\|\,\sum f_j(C_j)$ minimizing a generic regular function of job completion times (weighted tardiness in the computational study: $f_j(C_j) = w_j\max\{0,C_j-d_j\}$).

## Sets and Parameters

- $J = \{1,\ldots,n\}$ — set of jobs.

- $J_0 = J \cup \{0\}$ — jobs plus dummy job 0 with $p_0 = 0$.

- $m$ — number of identical parallel machines.

- $p_j$ — processing time of job $j$.

- $f_j(\cdot)$ — cost as a function of completion time.

- $T$ — time horizon (latest possible completion in an optimal schedule); a valid choice is $T = \lfloor (\sum_{j\in J} p_j - p_{\max})/m \rfloor + p_{\max}$ with $p_{\max} = \max_j p_j$.

## Decision Variables

Arc-time-indexed variables on a job-time graph: $$x^t_{ij} \in \mathbb{Z}_+$$ indicates that on some machine job $i$ finishes and job $j$ starts at time $t$ (so job $j$ completes at $t + p_j$). The variable $x^t_{00}$ represents a machine-idle arc at time $t$.

## Formulation

$$\begin{align}
\min\ & \sum_{i\in J_0}\ \sum_{j\in J\setminus\{i\}}\ \sum_{t=p_i}^{T-p_j}
       f_j(t+p_j)\, x^t_{ij} \tag{1a}\\[2pt]
\text{s.t.}\
& \sum_{j\in J_0} x^{0}_{0j} \;=\; m \tag{1b}\\[2pt]
& \sum_{i\in J_0\setminus\{j\}}\ \sum_{t=p_i}^{T-p_j} x^t_{ij} \;=\; 1
  && \forall\, j \in J \tag{1c}\\[2pt]
& \sum_{\substack{j\in J_0\setminus\{i\} \\ t-p_j\ge 0}} x^{t}_{ji}
  \;-\;
  \sum_{\substack{j\in J_0\setminus\{i\} \\ t+p_i\le T}} x^{t}_{ij}
  \;=\; 0
  && \forall\, i\in J,\ t=0,\ldots,T-p_i \tag{1d}\\[2pt]
& \sum_{\substack{j\in J_0 \\ t-p_j\ge 0}} x^{t}_{j0}
  \;-\;
  \sum_{\substack{j\in J_0 \\ t+p_j+1\le T}} x^{t+1}_{0j}
  \;=\; 0
  && t = 0,\ldots,T-1 \tag{1e}\\[2pt]
& x^t_{ij} \in \mathbb{Z}_+
  && \forall\, i\in J_0,\ j\in J_0\setminus\{i\},\ t=p_i,\ldots,T-p_j \tag{1f}\\
& x^t_{00} \in \mathbb{Z}_+
  && t = 0,\ldots,T-1 \tag{1g}
\end{align}$$

(1a) minimizes total completion-time cost. (1b) dispatches $m$ machines from the source node at time $0$. (1c) assigns every job exactly once. (1d) is flow conservation at each job-time node. (1e) is flow conservation for idle arcs through the dummy job. (1f)–(1g) are the integer domains.

## Released executable formulation and certificate

The public `gurobi_code.py` solves the same scheduling problem with a compact
disjunctive MILP (job-to-machine assignment, start times, and pairwise order)
instead of materializing every arc-time variable. This avoids a roughly
900,000-binary expansion on `tiny_instance_2.json`; it does not change the
feasible schedules or the weighted-tardiness objective. The released solution
certificate explicitly records `machine_id`, the immediate `prev_job`, start
and completion times, and each job's weighted-tardiness contribution. The
checker requires exact job coverage, verifies every named machine sequence and
the horizon, and recomputes the objective from the certificate.
