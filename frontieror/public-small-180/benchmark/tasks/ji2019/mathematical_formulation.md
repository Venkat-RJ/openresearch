# Sets and Indices

- $N$: set of ships, $n = |N|$.

- $M$: set of berths, $m = |M|$.

- $SL$: ships that can only pass through the lock.

- $SB$: ships that can only be transshipped.

- $Q$: set of lockages, $|Q| = n$.

- $MR_i \subseteq N$: ships to which ship $i \in N$ may moor.

- $P$: set of service orders per berth, $|P| = n$.

- $N(k) = \{i \in N : a_i \geq s_k\}$: ships arriving after berth $k$ is available.

- $B(i) \subseteq M$: berths that can serve ship $i$.

- $P(p) = \{l \in P : l < p\}$.

# Parameters

- $w_i, l_i$: width and length of ship $i$.

- $a_i$: arrival time of ship $i$.

- $W, L$: width and length of the lock chamber.

- $sc$: start-of-availability time of the chamber.

- $sp_0 = (0,0)$: chamber’s left side; $sp_{n+1} = (W, 0)$: chamber’s right side.

- $Du$: duration of a lockage; $MT$: minimal interval between consecutive lockages.

- $Pe_i$: penalty incurred by ship $i$ when it uses transshipment mode.

- $t_{ik}$: handling time of ship $i$ at berth $k$.

- $s_k, e_k$: availability start/end times of berth $k$.

- $C_{\max}$: upper bound on completion time (a big-$M$ constant).

# Decision Variables

- $u_i \in \{0,1\}$: mode chosen by ship $i$ ($1$ = transshipment, $0$ = lock).

- $x_i, y_i \in \mathbb{Z}_{\geq 0}$: $x$- and $y$-position of ship $i$ in a lockage.

- $lt_{ij} \in \{0,1\}$: $=1$ iff ship $i$ is entirely left of ship $j$.

- $b_{ij} \in \{0,1\}$: $=1$ iff ship $i$ is entirely behind ship $j$.

- $ml_{ij}, mr_{ij} \in \{0,1\}$: $=1$ iff ship $i$ moors on the left/right of ship $j$.

- $z_q \in \{0,1\}$: $=1$ iff lockage $q$ is used.

- $f_{iq} \in \{0,1\}$: $=1$ iff ship $i$ is transferred by lockage $q$.

- $v_{ij} \in \{0,1\}$: $=1$ iff ships $i$ and $j$ travel in the same lockage.

- $c_i \in [0, C_{\max}]$: departure time of ship $i$.

- $C_q \in [0, C_{\max}]$: completion time of lockage $q$.

- $\delta_{ipk} \in \{0,1\}$: $=1$ iff ship $i$ is assigned to the $p$-th order of berth $k$.

- $\omega_{ipk} \in \mathbb{Z}_{\geq 0}$: idle time of berth $k$ before ship $i$ arrives, when $i$ takes the $p$-th order of $k$.

# Objective

$$\begin{align}
\min\quad & T_1 + T_2 + \sum_{i \in N} Pe_i\, u_i, \tag{1}
\end{align}$$ where $$\begin{align}
T_1 &= \sum_{i \in N} (c_i - a_i), \tag{2}\\
T_2 &= \sum_{k \in M}\sum_{i \in N}\sum_{p \in P}
  \Bigl\{
    \sum_{j \in N}\sum_{l \in P(p)} \bigl(t_{jk}\,\delta_{jlk} + \omega_{jlk}\bigr)
    + \omega_{ipk} + t_{ik} + s_k - a_i
  \Bigr\}\,\delta_{ipk}. \tag{4}
\end{align}$$ (Constraint (3) is the nonlinear version of $T_2$, linearized as (4).)

# Constraints: Ship Placement Sub-problem

$$\begin{align}
& lt_{ij} + lt_{ji} + b_{ij} + b_{ji} + (1 - f_{iq}) + (1 - f_{jq}) \geq 1,
   && \forall i<j,\ i,j \in N,\ q \in Q, \tag{5}\\
& x_i + w_i \leq x_j + W(1 - lt_{ij}),
   && \forall i \neq j,\ i,j \in N, \tag{6}\\
& y_i + l_i \leq y_j + L(1 - b_{ij}),
   && \forall i \neq j,\ i,j \in N, \tag{7}\\
& x_i + w_i \leq W,
   && \forall i \in N, \tag{8}\\
& y_i + l_i \leq L,
   && \forall i \in N, \tag{9}\\
& \sum_{q \in Q} f_{iq} = 1 - u_i,
   && \forall i \in N, \tag{10}\\
& f_{iq} \leq z_q,
   && \forall i \in N,\ q \in Q, \tag{11}\\
& y_j - y_i \leq (1 - ml_{ij})\, L,
   && \forall i \in N,\ j \in MR_i, \tag{12}\\
& y_i - y_j \leq l_j - l_i + (1 - ml_{ij})\, L,
   && \forall i \in N,\ j \in MR_i, \tag{13}\\
& x_j - x_i \leq w_i + (1 - ml_{ij})\, W,
   && \forall i \in N,\ j \in MR_i, \tag{14}\\
& x_j - x_i \geq w_i - (1 - ml_{ij})\, W,
   && \forall i \in N,\ j \in MR_i, \tag{15}\\
& x_{n+1} - x_i \leq w_i + (1 - ml_{i,n+1})\, W,
   && \forall i \in N, \tag{16}\\
& x_{n+1} - x_i \geq w_i - (1 - ml_{i,n+1})\, W,
   && \forall i \in N, \tag{17}\\
& y_j - y_i \leq (1 - mr_{ij})\, L,
   && \forall i \in N,\ j \in MR_i, \tag{18}\\
& y_i - y_j \leq l_j - l_i + (1 - mr_{ij})\, L,
   && \forall i \in N,\ j \in MR_i, \tag{19}\\
& x_j - x_i \leq -\, w_j + (1 - mr_{ij})\, W,
   && \forall i \in N,\ j \in MR_i, \tag{20}\\
& x_j - x_i \geq -\, w_j - (1 - mr_{ij})\, W,
   && \forall i \in N,\ j \in MR_i, \tag{21}\\
& x_0 - x_i \leq -\, w_i + (1 - mr_{i,0})\, W,
   && \forall i \in N, \tag{22}\\
& x_0 - x_i \geq -\, w_i - (1 - mr_{i,0})\, W,
   && \forall i \in N, \tag{23}\\
& \sum_{j \in MR_i} (ml_{ij} + mr_{ij}) + mr_{i,0} + ml_{i,n+1}
     \geq 1 - u_i,
   && \forall i \in N, \tag{24}\\
& ml_{ij} + mr_{ji} \leq 1,
   && \forall i \in N,\ j \in MR_i, \tag{25}\\
& ml_{ij} + mr_{ij} \leq v_{ij},
   && \forall i \in N,\ j \in MR_i, \tag{26}\\
& f_{iq} - f_{jq} \leq 1 - v_{ij},
   && \forall i<j,\ i,j \in N,\ q \in Q, \tag{27}\\
& f_{jq} - f_{iq} \leq 1 - v_{ij},
   && \forall i<j,\ i,j \in N,\ q \in Q, \tag{28}\\
& v_{ij} \leq 1 - u_i,
   && \forall i<j,\ i,j \in N, \tag{29}\\
& v_{ij} \leq 1 - u_j,
   && \forall i<j,\ i,j \in N. \tag{30}
\end{align}$$

# Constraints: Lockage Scheduling Sub-problem

$$\begin{align}
& c_i \geq C_{\max}(f_{iq} - 1) + C_q,
   && \forall i \in N,\ q \in Q, \tag{31}\\
& c_i \leq C_{\max}(1 - f_{iq}) + C_q,
   && \forall i \in N,\ q \in Q, \tag{32}\\
& c_i \geq C_{\max}(u_i - 1) + a_i,
   && \forall i \in N, \tag{33}\\
& c_i \leq C_{\max}(1 - u_i) + a_i,
   && \forall i \in N, \tag{34}\\
& C_{q+1} - C_q \geq MT,
   && \forall q \in Q,\ q < |Q|, \tag{35}\\
& C_q - Du \geq sc,
   && \forall q \in Q, \tag{36}\\
& C_q - Du \geq f_{iq}\, a_i,
   && \forall i \in N,\ q \in Q, \tag{37}\\
& z_q \leq \sum_{i \in N} f_{iq},
   && \forall q \in Q, \tag{38}\\
& c_i - c_j \leq C_{\max}(u_i + u_j),
   && \forall i < j, \tag{39}\\
& z_{q+1} \leq z_q,
   && \forall q \in Q, \tag{40}\\
& C_q \leq C_{q+1},
   && \forall q \in Q, \tag{41}\\
& \sum_{q < l,\ q \in Q}(f_{iq} - f_{jq}) \geq -(u_i + u_j),
   && \forall i<j,\ i,j \in N,\ l \in Q. \tag{42}
\end{align}$$

# Mode Fixing

$$\begin{align}
& u_i = 0, && \forall i \in SL, \tag{43}\\
& u_i = 1, && \forall i \in SB. \tag{48}
\end{align}$$

# BAPD (Berth Allocation) Constraints

$$\begin{align}
& \sum_{k \in B(i)}\sum_{p \in P} \delta_{ipk} = u_i,
   && \forall i \in N, \tag{44}\\
& \sum_{i \in N} \delta_{ipk} \leq 1,
   && \forall k \in M,\ p \in P, \tag{45}\\
& \sum_{j \in N}\sum_{l \in P(p)}\bigl(t_{jk}\,\delta_{jlk} + \omega_{jlk}\bigr)
   + \omega_{ipk} - (a_i - s_k)\,\delta_{ipk} \geq 0,
   && \forall k \in M,\ i \in N(k),\ p \in P, \tag{46}\\
& \sum_{i \in N}\sum_{p \in P}\bigl(t_{ik}\,\delta_{ipk} + \omega_{ipk}\bigr)
   \leq e_k - s_k,
   && \forall k \in M. \tag{47}
\end{align}$$

# Integrality and Bounds

$$\begin{align}
& u_i \in \{0,1\},
   && \forall i \in N, \tag{49}\\
& lt_{ij},\ b_{ij} \in \{0,1\},
   && \forall i \neq j,\ i,j \in N, \tag{50}\\
& ml_{ij},\ mr_{ij} \in \{0,1\},
   && \forall i \in N,\ j \in MR_i, \tag{51}\\
& ml_{i,n+1},\ mr_{i,0} \in \{0,1\},
   && \forall i \in N, \tag{52}\\
& v_{ij} \in \{0,1\},\quad z_q \in \{0,1\},
   && \forall i<j,\ q \in Q, \tag{53}\\
& 0 \leq c_i \leq C_{\max},\quad 0 \leq C_q \leq C_{\max},
   && \forall i \in N,\ q \in Q, \tag{54}\\
& f_{iq} \in \{0,1\},
   && \forall i \in N,\ q \in Q, \tag{55}\\
& x_i, y_i \in \mathbb{Z}_{\geq 0},
   && \forall i \in N, \tag{56}\\
& \delta_{ipk} \in \{0,1\},\quad \omega_{ipk} \geq 0,
   && \forall i \in N,\ k \in M,\ p \in P. \tag{57}
\end{align}$$
