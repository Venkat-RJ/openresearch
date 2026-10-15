#!/usr/bin/env python3
"""
TDTSP (Time-Dependent Travelling Salesman Problem) Branch-and-Cut Solver
Based on: Cordeau, Ghiani, and Guerriero (2014)
          "Analysis and Branch-and-Cut Algorithm for the Time-Dependent
           Travelling Salesman Problem"
          Transportation Science 48(1), pp. 46-58.

This implementation uses Gurobi with lazy constraint callbacks for
subtour elimination (16) and path-based lower bound constraints (13).
"""

import argparse
import heapq
import json
import math
import sys
import time
from collections import defaultdict
from itertools import combinations

import gurobipy as gp
from gurobipy import GRB
import os as _os, sys as _sys
# Walk up from this file's directory to find repo root (containing scripts/).
_GUROBI_CODE_START_TIME = time.time()
_repo = _os.path.dirname(_os.path.abspath(__file__))
while _repo != _os.path.dirname(_repo) and not _os.path.isdir(_os.path.join(_repo, 'scripts', 'utils')):
    _repo = _os.path.dirname(_repo)
if _os.path.isdir(_os.path.join(_repo, 'scripts', 'utils')):
    _sys.path.insert(0, _repo)
try:
    from scripts.utils.gurobi_log_helper import install_gurobi_logger
except ImportError:
    def install_gurobi_logger(log_path):  # no-op fallback when scripts/ unavailable
        pass


# ============================================================================
# Algorithm 1: Travel time computation
# ============================================================================

def compute_travel_time(i, j, t0, L, v_ijh, period_boundaries, H):
    """
    Algorithm 1 from the paper.
    Compute travel time on arc (i,j) departing at time t0.

    Parameters
    ----------
    i, j : int
        Origin and destination vertex indices.
    t0 : float
        Departure time.
    L : 2D list/array
        Distance matrix L[i][j].
    v_ijh : 3D list/array
        Speed array v_ijh[i][j][h].
    period_boundaries : list
        Period boundary times [T_0, T_1, ..., T_H].
    H : int
        Number of time periods.

    Returns
    -------
    float
        Travel time tau_{ij}(t0).
    """
    if i == j:
        return 0.0

    dist = L[i][j]
    if dist <= 0:
        return 0.0

    # Find period k0 such that T_{k0} <= t0 <= T_{k0+1}
    k = 0
    for h in range(H):
        if period_boundaries[h] <= t0 <= period_boundaries[h + 1]:
            k = h
            break
    else:
        # t0 is at or beyond the last boundary; use the last period
        k = H - 1

    t = t0
    d = dist
    speed = v_ijh[i][j][k]
    if speed <= 1e-12:
        # Speed is essentially zero; return a very large travel time
        return 1e15

    t_prime = t + d / speed

    while t_prime > period_boundaries[k + 1] + 1e-12 and k < H - 1:
        # Distance covered in current period k
        d = d - speed * (period_boundaries[k + 1] - t)
        t = period_boundaries[k + 1]
        k = k + 1
        speed = v_ijh[i][j][k]
        if speed <= 1e-12:
            return 1e15
        t_prime = t + d / speed

    return t_prime - t0


def compute_travel_time_common_congestion(i, j, t0, L, u_ij, b_h,
                                          period_boundaries, H):
    """
    Compute travel time on arc (i,j) under common congestion model:
    speed = b_h * u_{ij} in each period h.

    Uses Algorithm 1 logic with v_{ijh} = b_h * u_{ij}.
    """
    if i == j:
        return 0.0

    dist = L[i][j]
    if dist <= 0:
        return 0.0

    u = u_ij[i][j]
    if u <= 1e-12:
        return 1e15

    # Find period k0
    k = 0
    for h in range(H):
        if period_boundaries[h] <= t0 <= period_boundaries[h + 1]:
            k = h
            break
    else:
        k = H - 1

    t = t0
    d = dist
    speed = b_h[k] * u
    if speed <= 1e-12:
        return 1e15

    t_prime = t + d / speed

    while t_prime > period_boundaries[k + 1] + 1e-12 and k < H - 1:
        d = d - speed * (period_boundaries[k + 1] - t)
        t = period_boundaries[k + 1]
        k = k + 1
        speed = b_h[k] * u
        if speed <= 1e-12:
            return 1e15
        t_prime = t + d / speed

    return t_prime - t0


def compute_dummy_travel_time(length, start_time, b_h, period_boundaries, H):
    """
    Compute travel time of a 'dummy arc' with given length and speed b_h
    in each period h. Used in Algorithm 2 (LB_p computation).

    This is Algorithm 1 with L = length and v_h = b_h for all h.
    """
    if length <= 1e-12:
        return 0.0

    # Find period k0
    k = 0
    for h in range(H):
        if period_boundaries[h] <= start_time <= period_boundaries[h + 1]:
            k = h
            break
    else:
        k = H - 1

    t = start_time
    d = length
    speed = b_h[k]
    if speed <= 1e-12:
        return 1e15

    t_prime = t + d / speed

    while t_prime > period_boundaries[k + 1] + 1e-12 and k < H - 1:
        d = d - speed * (period_boundaries[k + 1] - t)
        t = period_boundaries[k + 1]
        k = k + 1
        speed = b_h[k]
        if speed <= 1e-12:
            return 1e15
        t_prime = t + d / speed

    return t_prime - start_time


# ============================================================================
# Tour evaluation functions
# ============================================================================

def evaluate_tour_tdtsp(tour, L, v_ijh, period_boundaries, H):
    """
    Evaluate a tour under the actual TDTSP speed model.
    tour: list of vertices starting and ending at depot 0.
    Returns z(c) = total tour duration.
    """
    current_time = 0.0
    for idx in range(len(tour) - 1):
        i = tour[idx]
        j = tour[idx + 1]
        tt = compute_travel_time(i, j, current_time, L, v_ijh,
                                 period_boundaries, H)
        current_time += tt
    return current_time


def evaluate_tour_common_congestion(tour, L, u_ij, b_h,
                                    period_boundaries, H):
    """
    Evaluate a tour under the common congestion speed model:
    v_{ijh} = b_h * u_{ij}.
    Returns z_underline(c).
    """
    current_time = 0.0
    for idx in range(len(tour) - 1):
        i = tour[idx]
        j = tour[idx + 1]
        tt = compute_travel_time_common_congestion(i, j, current_time, L,
                                                   u_ij, b_h,
                                                   period_boundaries, H)
        current_time += tt
    return current_time


def evaluate_tour_atsp(tour, L, u_ij):
    """
    Evaluate a tour under ATSP costs: L_{ij}/u_{ij}.
    Returns z_bar_bar(c).
    """
    total = 0.0
    for idx in range(len(tour) - 1):
        i = tour[idx]
        j = tour[idx + 1]
        if u_ij[i][j] > 1e-12:
            total += L[i][j] / u_ij[i][j]
    return total


# ============================================================================
# Path evaluation for LB_p (Algorithm 2)
# ============================================================================

def evaluate_path_tdtsp(path, L, v_ijh, period_boundaries, H):
    """
    Evaluate actual TDTSP traversal time of a path starting at time 0.
    Returns z(p).
    """
    current_time = 0.0
    for idx in range(len(path) - 1):
        i = path[idx]
        j = path[idx + 1]
        tt = compute_travel_time(i, j, current_time, L, v_ijh,
                                 period_boundaries, H)
        current_time += tt
    return current_time


def evaluate_path_atsp(path, L, u_ij):
    """
    Evaluate ATSP cost of a path: sum of L_{ab}/u_{ab} for (a,b) in path.
    Returns z_bar_bar(p).
    """
    total = 0.0
    for idx in range(len(path) - 1):
        a = path[idx]
        b = path[idx + 1]
        if u_ij[a][b] > 1e-12:
            total += L[a][b] / u_ij[a][b]
    return total


def compute_LB_p(path, atsp_lower_bound, L, u_ij, v_ijh, b_h,
                 period_boundaries, H):
    """
    Algorithm 2: Compute lower bound LB_p for a path p.

    Parameters
    ----------
    path : list of int
        Simple path from depot 0 to some vertex i.
    atsp_lower_bound : float
        Certified lower bound on the optimal ATSP value.
    L, u_ij, v_ijh, b_h, period_boundaries, H : instance data.

    Returns
    -------
    float
        LB_p.
    """
    z_p = evaluate_path_tdtsp(path, L, v_ijh, period_boundaries, H)
    zbar_p = evaluate_path_atsp(path, L, u_ij)

    if atsp_lower_bound < zbar_p:
        return z_p
    else:
        remaining_length = atsp_lower_bound - zbar_p
        tau_val = compute_dummy_travel_time(remaining_length, z_p, b_h,
                                           period_boundaries, H)
        return z_p + tau_val


# ============================================================================
# ATSP solver (Step 1)
# ============================================================================

def solve_atsp(n_total, L, u_ij, time_limit=600):
    """
    Solve the ATSP with arc costs L_{ij}/u_{ij} using Gurobi with
    lazy subtour elimination constraints.

    Parameters
    ----------
    n_total : int
        Total number of vertices (n_customers + 1, including depot 0).
    L : 2D list
        Distance matrix.
    u_ij : 2D list
        Max speed matrix.
    time_limit : int
        Time limit in seconds.

    Returns
    -------
    tour : list of int
        Optimal tour as a sequence of vertices (starting and ending at 0).
    atsp_incumbent : float
        Best ATSP tour value found.
    atsp_bound : float
        Certified lower bound on the ATSP optimum.
    atsp_status : int
        Gurobi termination status for the ATSP subproblem.
    """
    vertices = list(range(n_total))
    arcs = [(i, j) for i in vertices for j in vertices if i != j]

    # Arc costs
    cost = {}
    for i, j in arcs:
        if u_ij[i][j] > 1e-12:
            cost[i, j] = L[i][j] / u_ij[i][j]
        else:
            cost[i, j] = 1e15  # effectively prohibit

    model = gp.Model("ATSP")
    model.setParam("Threads", 1)
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = time_limit
    model.Params.LazyConstraints = 1

    # Binary variables
    x = model.addVars(arcs, vtype=GRB.BINARY, name="x")

    # Objective
    model.setObjective(gp.quicksum(cost[i, j] * x[i, j]
                                   for i, j in arcs), GRB.MINIMIZE)

    # Degree constraints
    for j in vertices:
        model.addConstr(
            gp.quicksum(x[i, j] for i in vertices if i != j) == 1,
            name=f"indeg_{j}")
    for i in vertices:
        model.addConstr(
            gp.quicksum(x[i, j] for j in vertices if j != i) == 1,
            name=f"outdeg_{i}")

    def subtour_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            x_val = model.cbGetSolution(x)
            # Build adjacency from solution
            adj = {}
            for i, j in arcs:
                if x_val[i, j] > 0.5:
                    adj[i] = j

            # Find connected components (subtours)
            visited = set()
            subtours = []
            for start in vertices:
                if start in visited:
                    continue
                tour = []
                current = start
                while current not in visited:
                    visited.add(current)
                    tour.append(current)
                    current = adj.get(current, start)
                if len(tour) < n_total:
                    subtours.append(tour)

            # Add SEC for each subtour not containing all vertices
            for S in subtours:
                if len(S) < n_total:
                    S_set = set(S)
                    model.cbLazy(
                        gp.quicksum(x[i, j]
                                    for i in S_set
                                    for j in vertices
                                    if j not in S_set and i != j) >= 1)

    model.optimize(subtour_callback)

    if model.Status == GRB.OPTIMAL or model.SolCount > 0:
        # Extract tour
        adj = {}
        for i, j in arcs:
            if x[i, j].X > 0.5:
                adj[i] = j

        tour = [0]
        current = adj[0]
        while current != 0:
            tour.append(current)
            current = adj[current]
        tour.append(0)

        return tour, model.ObjVal, model.ObjBound, model.Status
    else:
        print("ATSP solve failed. Status:", model.Status)
        sys.exit(1)


# ============================================================================
# Shortest path for constraint (13) separation
# ============================================================================

def dijkstra_from_source(source, n_total, arc_costs):
    """
    Dijkstra's algorithm from a single source with given arc costs.
    Returns (dist, pred) where dist[v] is shortest distance and
    pred[v] is the predecessor on the shortest path.
    """
    INF = float('inf')
    dist = [INF] * n_total
    pred = [-1] * n_total
    dist[source] = 0.0
    visited = [False] * n_total
    pq = [(0.0, source)]

    while pq:
        d, u = heapq.heappop(pq)
        if visited[u]:
            continue
        visited[u] = True
        for v in range(n_total):
            if v == u:
                continue
            w = arc_costs.get((u, v), INF)
            if d + w < dist[v]:
                dist[v] = d + w
                pred[v] = u
                heapq.heappush(pq, (dist[v], v))

    return dist, pred


def reconstruct_path(pred, target):
    """Reconstruct path from source to target using predecessor array."""
    path = []
    v = target
    while v != -1:
        path.append(v)
        v = pred[v]
    path.reverse()
    return path


# ============================================================================
# Connected components for subtour detection
# ============================================================================

def find_connected_components(n_total, edges):
    """
    Find connected components in an undirected graph.
    edges: list of (i, j, weight) tuples.
    Returns list of sets, each set being a connected component.
    """
    adj = defaultdict(set)
    active_nodes = set()
    for i, j, w in edges:
        if w > 1e-6:
            adj[i].add(j)
            adj[j].add(i)
            active_nodes.add(i)
            active_nodes.add(j)

    visited = set()
    components = []
    for start in range(n_total):
        if start in visited:
            continue
        if start not in active_nodes:
            # Isolated node; add as single component
            visited.add(start)
            components.append({start})
            continue
        # BFS
        comp = set()
        queue = [start]
        while queue:
            node = queue.pop()
            if node in visited:
                continue
            visited.add(node)
            comp.add(node)
            for neighbor in adj[node]:
                if neighbor not in visited:
                    queue.append(neighbor)
        components.append(comp)

    return components


def find_subtours_integer(n_total, x_vals):
    """
    Given integer x values, find subtours by following arcs.
    Returns list of subtours (each a list of vertices).
    """
    adj = {}
    for (i, j), val in x_vals.items():
        if val > 0.5:
            adj[i] = j

    visited = set()
    subtours = []
    for start in range(n_total):
        if start in visited:
            continue
        tour = []
        current = start
        while current not in visited:
            visited.add(current)
            tour.append(current)
            if current not in adj:
                break
            current = adj[current]
        subtours.append(tour)

    return subtours


def find_min_cut_subtour(n_total, x_vals):
    """
    Find violated subtour elimination constraints from fractional solution.
    Uses a simple approach: build undirected support graph, find min cut
    via connected components for integer solutions, or use flow-based
    approach for fractional.

    For fractional solutions, we check all subsets S containing depot 0
    and look for x(S, V\\S) < 1. In practice, we use a max-flow / min-cut
    heuristic.

    Returns list of violated subsets S (each a set of vertex indices).
    """
    # Build undirected support graph capacity
    capacity = defaultdict(float)
    for (i, j), val in x_vals.items():
        if val > 1e-6:
            capacity[i, j] += val

    violated = []

    # For each vertex t != 0, compute min s-t cut from 0 to t
    for t in range(1, n_total):
        # Use BFS-based max-flow (Edmonds-Karp)
        flow_val, S_set = max_flow_min_cut(0, t, n_total, capacity)
        if flow_val < 1.0 - 1e-6:
            violated.append(S_set)

    return violated


def max_flow_min_cut(s, t, n, capacity):
    """
    Edmonds-Karp (BFS-based) max-flow algorithm.
    Returns (max_flow_value, S_set) where S_set is the source side
    of the min cut.
    """
    # Build residual graph
    residual = defaultdict(float)
    for (i, j), cap in capacity.items():
        residual[i, j] += cap

    total_flow = 0.0

    while True:
        # BFS to find augmenting path
        parent = {s: None}
        visited = {s}
        queue = [s]
        found = False
        while queue and not found:
            u = queue.pop(0)
            for v in range(n):
                if v not in visited and residual.get((u, v), 0) > 1e-10:
                    parent[v] = u
                    visited.add(v)
                    if v == t:
                        found = True
                        break
                    queue.append(v)

        if not found:
            break

        # Find bottleneck
        bottleneck = float('inf')
        v = t
        while parent[v] is not None:
            u = parent[v]
            bottleneck = min(bottleneck, residual[u, v])
            v = u
        if bottleneck <= 1e-10:
            break

        # Update residual
        v = t
        while parent[v] is not None:
            u = parent[v]
            residual[u, v] -= bottleneck
            residual[v, u] += bottleneck
            v = u

        total_flow += bottleneck

    # Find S-side of min cut (reachable from s in residual)
    S_set = set()
    visited_final = {s}
    queue = [s]
    while queue:
        u = queue.pop(0)
        S_set.add(u)
        for v in range(n):
            if v not in visited_final and residual.get((u, v), 0) > 1e-10:
                visited_final.add(v)
                queue.append(v)

    return total_flow, S_set


# ============================================================================
# Main TDTSP Branch-and-Cut Model
# ============================================================================

def solve_tdtsp(instance_path, solution_path, time_limit):
    """
    Main function: load instance, solve ATSP for initialization,
    build and solve the TDTSP branch-and-cut model.
    """
    # ------------------------------------------------------------------
    # 1. Load instance data
    # ------------------------------------------------------------------
    with open(instance_path, 'r') as f:
        data = json.load(f)

    params = data["parameters"]
    n_customers = params["n_customers"]
    n_total = params["total_vertices"]  # n_customers + 1 (depot)
    H = params["H"]
    b_h = params["b_h"]
    T = params["time_horizon_T"]
    period_boundaries = params["period_boundaries"]

    L = data["arcs"]["distances"]
    u_ij = data["arcs"]["max_speeds_u_ij"]
    v_ijh = data["arcs"]["speeds_v_ijh"]

    vertices = list(range(n_total))
    arcs = [(i, j) for i in vertices for j in vertices if i != j]

    # Precompute ATSP arc costs: L_{ij}/u_{ij}
    atsp_cost = {}
    for i, j in arcs:
        if u_ij[i][j] > 1e-12:
            atsp_cost[i, j] = L[i][j] / u_ij[i][j]
        else:
            atsp_cost[i, j] = 1e15

    print("=" * 70)
    print("TDTSP Branch-and-Cut Solver")
    print(f"Instance: {instance_path}")
    print(f"Vertices: {n_total} (depot + {n_customers} customers)")
    print(f"Time periods: {H}, b_h = {b_h}")
    print(f"Time horizon: {T:.6f}")
    print(f"Period boundaries: {period_boundaries}")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 2. Solve ATSP to get optimal tour c*
    # ------------------------------------------------------------------
    print("\n--- Step 1: Solving ATSP ---")
    atsp_time_limit = max(60, time_limit // 4)
    atsp_tour, atsp_incumbent, atsp_bound, atsp_status = solve_atsp(
        n_total, L, u_ij, atsp_time_limit
    )
    print(f"ATSP incumbent tour: {atsp_tour}")
    print(f"ATSP incumbent value = {atsp_incumbent:.6f}")
    print(f"ATSP certified lower bound = {atsp_bound:.6f}")

    # ------------------------------------------------------------------
    # 3. Compute initial bounds
    # ------------------------------------------------------------------
    print("\n--- Step 2: Computing initial bounds ---")

    # z_bar_bar(c*) = ATSP value (already computed)
    # Path lower bounds must use a certified bound.  Treating a time-limited
    # incumbent as the ATSP optimum can invalidate every downstream cut.
    zbarbar_cstar = atsp_bound

    # z(c*) = evaluate tour under actual TDTSP speeds (upper bound)
    z_cstar = evaluate_tour_tdtsp(atsp_tour, L, v_ijh, period_boundaries, H)
    print(f"TDTSP evaluation z(c*) = {z_cstar:.6f} (upper bound)")

    # z_underline(c*) = evaluate tour under common congestion (lower bound)
    # Under common congestion, normalized arc lengths L_ij/u_ij are traversed
    # at the shared speed b_h.  Applying that time transform to the certified
    # ATSP bound yields a valid initial lower bound even when the ATSP
    # subproblem stops with only an incumbent.
    z_underline_cstar = compute_dummy_travel_time(
        atsp_bound, 0.0, b_h, period_boundaries, H
    )
    print(f"Certified common-congestion lower bound = "
          f"{z_underline_cstar:.6f}")

    initial_gap = (z_cstar - z_underline_cstar) / z_cstar * 100
    print(f"Initial gap: {initial_gap:.2f}%")

    # ------------------------------------------------------------------
    # 4. Build the main TDTSP ILP model
    # ------------------------------------------------------------------
    print("\n--- Step 3: Building TDTSP ILP model ---")

    model = gp.Model("TDTSP")
    model.setParam("Threads", 1)
    model.Params.OutputFlag = 1
    model.Params.TimeLimit = time_limit
    model.Params.LazyConstraints = 1
    # Use Gurobi's default settings for other parameters

    # --- Decision Variables ---

    # x_{ij} binary arc variables
    x = model.addVars(arcs, vtype=GRB.BINARY, name="x")

    # z: total tour duration (continuous)
    z = model.addVar(vtype=GRB.CONTINUOUS, lb=0.0, name="z")

    # z_h: auxiliary variables for valid inequalities, h = 0,...,H-1
    z_h = model.addVars(H, vtype=GRB.CONTINUOUS, lb=0.0, name="z_h")

    # y_h: binary variables, h = 0,...,H-1
    y_h = model.addVars(H, vtype=GRB.BINARY, name="y_h")

    # --- Objective (12): minimize z ---
    model.setObjective(z, GRB.MINIMIZE)

    # --- Degree constraints (14): in-degree ---
    for j in vertices:
        model.addConstr(
            gp.quicksum(x[i, j] for i in vertices if i != j) == 1,
            name=f"indeg_{j}")

    # --- Degree constraints (15): out-degree ---
    for i in vertices:
        model.addConstr(
            gp.quicksum(x[i, j] for j in vertices if j != i) == 1,
            name=f"outdeg_{i}")

    # --- Initial lower bound (31): z >= z_underline(c*) ---
    model.addConstr(z >= z_underline_cstar, name="init_lb")

    # --- Set initial upper bound via cutoff ---
    model.Params.Cutoff = z_cstar + 1e-6

    # --- Provide the ATSP tour as a MIP start ---
    for i, j in arcs:
        x[i, j].Start = 0.0
    for idx in range(len(atsp_tour) - 1):
        ai = atsp_tour[idx]
        aj = atsp_tour[idx + 1]
        x[ai, aj].Start = 1.0

    # --- Valid inequalities (20)-(27) ---

    # Precompute: ATSP_sum = sum of (L_{ij}/u_{ij}) * x_{ij} over all arcs
    atsp_sum_expr = gp.quicksum(atsp_cost[i, j] * x[i, j]
                                for i, j in arcs)

    # (20) z >= sum_{h=0}^{H-1} z_h / b_h
    model.addConstr(
        z >= gp.quicksum(z_h[h] / b_h[h] for h in range(H)),
        name="vi_20")

    # (21) z_0 >= ATSP_sum - sum_{l=1}^{H-1} z_l
    if H >= 1:
        model.addConstr(
            z_h[0] >= atsp_sum_expr
            - gp.quicksum(z_h[l] for l in range(1, H)),
            name="vi_21")

    # (22) z_h >= ATSP_sum - sum_{l=0}^{h-1} b_l*(T_{l+1}-T_l)
    #                       - sum_{l=h+1}^{H-1} z_l
    #      for h = 1, ..., H-2
    for h in range(1, H - 1):
        sum_periods = sum(b_h[l] * (period_boundaries[l + 1]
                                    - period_boundaries[l])
                          for l in range(h))
        model.addConstr(
            z_h[h] >= atsp_sum_expr - sum_periods
            - gp.quicksum(z_h[l] for l in range(h + 1, H)),
            name=f"vi_22_{h}")

    # (23) z_{H-1} >= ATSP_sum - sum_{l=0}^{H-2} b_l*(T_{l+1}-T_l)
    if H >= 2:
        sum_periods_all = sum(b_h[l] * (period_boundaries[l + 1]
                                        - period_boundaries[l])
                              for l in range(H - 1))
        model.addConstr(
            z_h[H - 1] >= atsp_sum_expr - sum_periods_all,
            name="vi_23")

    # (24) z_h <= b_h*(T_{h+1}-T_h) * sum_{l=h}^{H-1} y_l
    #      for h = 0,...,H-1
    for h in range(H):
        period_len = period_boundaries[h + 1] - period_boundaries[h]
        model.addConstr(
            z_h[h] <= b_h[h] * period_len
            * gp.quicksum(y_h[l] for l in range(h, H)),
            name=f"vi_24_{h}")

    # (25) z_h >= b_h*(T_{h+1}-T_h) * sum_{l=h+1}^{H-1} y_l
    #      for h = 0,...,H-2
    for h in range(H - 1):
        period_len = period_boundaries[h + 1] - period_boundaries[h]
        model.addConstr(
            z_h[h] >= b_h[h] * period_len
            * gp.quicksum(y_h[l] for l in range(h + 1, H)),
            name=f"vi_25_{h}")

    # (26) sum_{h=0}^{H-1} y_h = 1
    model.addConstr(
        gp.quicksum(y_h[h] for h in range(H)) == 1,
        name="vi_26")

    # (27) z_h >= 0 (already set as lb=0), y_h binary (already set)

    # ------------------------------------------------------------------
    # 5. Callback for lazy constraints
    # ------------------------------------------------------------------
    callback_data = {
        'n_total': n_total,
        'vertices': vertices,
        'arcs': arcs,
        'L': L,
        'u_ij': u_ij,
        'v_ijh': v_ijh,
        'b_h': b_h,
        'period_boundaries': period_boundaries,
        'H': H,
        'atsp_lower_bound': zbarbar_cstar,
        'atsp_cost': atsp_cost,
        'x': x,
        'z': z,
        'sec_count': 0,
        'path_count': 0,
        'best_obj': z_cstar,
    }

    def tdtsp_callback(model, where):
        if where == GRB.Callback.MIPSOL:
            # Integer solution found: check for subtours and update bound
            _separate_integer(model, callback_data)
        elif where == GRB.Callback.MIPNODE:
            # At a node: separate fractional cuts
            status = model.cbGet(GRB.Callback.MIPNODE_STATUS)
            if status == GRB.OPTIMAL:
                _separate_fractional(model, callback_data)

    def _separate_integer(model, cb):
        """Separate subtour elimination at integer solutions."""
        x = cb['x']
        z_var = cb['z']
        n = cb['n_total']
        verts = cb['vertices']

        x_val = {}
        for i, j in cb['arcs']:
            x_val[i, j] = model.cbGetSolution(x[i, j])

        z_val = model.cbGetSolution(z_var)

        # Check subtours
        subtours = find_subtours_integer(n, x_val)

        if len(subtours) > 1:
            # Multiple subtours found: add SEC for each
            for S in subtours:
                if len(S) < n:
                    S_set = set(S)
                    model.cbLazy(
                        gp.quicksum(x[i, j]
                                    for i in S_set
                                    for j in verts
                                    if j not in S_set and i != j) >= 1)
                    cb['sec_count'] += 1
        else:
            # Single tour (Hamiltonian): check if we can update best
            # Extract the tour
            adj = {}
            for (i, j), val in x_val.items():
                if val > 0.5:
                    adj[i] = j
            tour = [0]
            current = adj.get(0, 0)
            while current != 0 and len(tour) <= n:
                tour.append(current)
                current = adj.get(current, 0)
            tour.append(0)

            # Evaluate actual TDTSP value
            actual_z = evaluate_tour_tdtsp(
                tour, cb['L'], cb['v_ijh'],
                cb['period_boundaries'], cb['H'])

            # The constraint z >= actual_z should hold for this tour
            if actual_z > z_val + 1e-6:
                # The current z is too low; add path constraint for this tour
                # We add it as a constraint on the full tour
                num_arcs = len(tour) - 1
                model.cbLazy(
                    z_var >= actual_z * (
                        gp.quicksum(
                            x[tour[idx], tour[idx + 1]]
                            for idx in range(num_arcs)
                        ) - num_arcs + 1
                    ))
                cb['path_count'] += 1

            if actual_z < cb['best_obj']:
                cb['best_obj'] = actual_z

    def _separate_fractional(model, cb):
        """Separate subtour elimination and path constraints at
        fractional nodes."""
        x = cb['x']
        z_var = cb['z']
        n = cb['n_total']
        verts = cb['vertices']

        x_val = {}
        for i, j in cb['arcs']:
            x_val[i, j] = model.cbGetNodeRel(x[i, j])

        z_val = model.cbGetNodeRel(z_var)

        # --- Subtour elimination separation ---
        # Build directed capacity for min-cut
        capacity = defaultdict(float)
        for (i, j), val in x_val.items():
            if val > 1e-6:
                capacity[i, j] = val

        sec_found = False
        for t in range(1, n):
            flow_val, S_set = max_flow_min_cut(0, t, n, capacity)
            if flow_val < 1.0 - 1e-4:
                # Violated SEC
                S_bar = set(verts) - S_set
                model.cbCut(
                    gp.quicksum(x[i, j]
                                for i in S_set
                                for j in S_bar
                                if i != j and (i, j) in x) >= 1)
                cb['sec_count'] += 1
                sec_found = True

        if sec_found:
            return  # Try SECs again in next round

        # --- Path constraint (13) separation ---
        # Arc costs: (1 - x_hat_{ij})
        arc_costs_sp = {}
        for i, j in cb['arcs']:
            arc_costs_sp[i, j] = max(0.0, 1.0 - x_val[i, j])

        # Dijkstra from vertex 0
        dist, pred = dijkstra_from_source(0, n, arc_costs_sp)

        path_found = False
        for target in range(1, n):
            if dist[target] < 1.0 - 1e-6:
                # Violated path constraint
                path = reconstruct_path(pred, target)

                if len(path) < 2:
                    continue

                # Compute LB_p
                LB_p = compute_LB_p(
                    path, cb['atsp_lower_bound'],
                    cb['L'], cb['u_ij'], cb['v_ijh'],
                    cb['b_h'], cb['period_boundaries'], cb['H'])

                if LB_p <= 1e-6:
                    continue

                # Constraint (13):
                # z >= LB_p * (1 - sum_{(a,b) in p} (1 - x_{ab}))
                # = LB_p * (sum x_{ab} - m + 1), m = #arcs in path.
                # It is active when every path arc is selected and relaxes as
                # soon as an arc is absent.
                num_arcs_in_path = len(path) - 1
                rhs_val = LB_p * (
                    sum(x_val[path[k], path[k + 1]]
                        for k in range(num_arcs_in_path))
                    - num_arcs_in_path + 1
                )

                if rhs_val > z_val + 1e-6:
                    path_arcs_sum = gp.quicksum(
                        x[path[k], path[k + 1]]
                        for k in range(num_arcs_in_path))
                    model.cbCut(
                        z_var >= LB_p * (
                            path_arcs_sum - num_arcs_in_path + 1))
                    cb['path_count'] += 1
                    path_found = True

        # If path constraints were found, the next iteration will
        # re-examine SECs

    # ------------------------------------------------------------------
    # 6. Optimize
    # ------------------------------------------------------------------
    print("\n--- Step 4: Solving TDTSP model ---")
    start_time = time.time()
    model.optimize(tdtsp_callback)
    solve_time = time.time() - start_time

    print(f"\nSolve time: {solve_time:.2f} seconds")
    print(f"SECs added: {callback_data['sec_count']}")
    print(f"Path constraints added: {callback_data['path_count']}")

    # ------------------------------------------------------------------
    # 7. Extract solution
    # ------------------------------------------------------------------
    result = {
        "instance_path": instance_path,
        "solver": "gurobi",
        "model": "TDTSP_BranchAndCut",
        "time_limit": time_limit,
        "solve_time": solve_time,
        "status": model.Status,
        "status_name": _status_name(model.Status),
        "atsp_status": atsp_status,
        "atsp_status_name": _status_name(atsp_status),
        "atsp_incumbent": atsp_incumbent,
        "atsp_certified_bound": atsp_bound,
    }

    if model.SolCount > 0:
        # Extract tour from x values
        adj = {}
        for i, j in arcs:
            if x[i, j].X > 0.5:
                adj[i] = j

        tour = [0]
        current = adj.get(0, -1)
        max_steps = n_total + 1
        steps = 0
        while current != 0 and current != -1 and steps < max_steps:
            tour.append(current)
            current = adj.get(current, -1)
            steps += 1
        tour.append(0)

        # Model objective (z variable value)
        model_obj = z.X

        # Evaluate actual TDTSP tour time
        actual_obj = evaluate_tour_tdtsp(
            tour, L, v_ijh, period_boundaries, H)

        # The true objective is the actual TDTSP evaluation of the best tour.
        # Use the better of the MIP solution tour and the initial ATSP tour.
        if actual_obj <= z_cstar:
            final_obj = actual_obj
        else:
            final_obj = z_cstar
            tour = atsp_tour

        result.update({
            "objective_value": final_obj,
            "model_z_value": model_obj,
            "actual_tdtsp_value": actual_obj,
            "atsp_upper_bound": z_cstar,
            "initial_lower_bound": z_underline_cstar,
            "tour": tour,
            "n_customers": n_customers,
            "n_vertices": n_total,
            "mip_gap": model.MIPGap if hasattr(model, 'MIPGap') else None,
            "best_bound": model.ObjBound if hasattr(model, 'ObjBound')
            else None,
            "node_count": int(model.NodeCount),
        })

        print(f"\nModel z value: {model_obj:.6f}")
        print(f"Actual TDTSP evaluation: {actual_obj:.6f}")
        print(f"ATSP heuristic UB: {z_cstar:.6f}")
        print(f"Final objective: {final_obj:.6f}")
        print(f"Tour: {tour}")
        if hasattr(model, 'MIPGap'):
            try:
                print(f"MIP gap: {model.MIPGap * 100:.4f}%")
            except Exception:
                pass
    else:
        # No feasible solution found; use ATSP heuristic
        print("\nNo MIP solution found. Using ATSP heuristic solution.")
        result.update({
            "objective_value": z_cstar,
            "model_z_value": None,
            "actual_tdtsp_value": z_cstar,
            "atsp_upper_bound": z_cstar,
            "initial_lower_bound": z_underline_cstar,
            "tour": atsp_tour,
            "n_customers": n_customers,
            "n_vertices": n_total,
            "mip_gap": None,
            "best_bound": None,
            "node_count": 0,
        })

    # ------------------------------------------------------------------
    # 8. Write solution
    # ------------------------------------------------------------------
    with open(solution_path, 'w') as f:
        result["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(result, f, indent=2)
    print(f"\nSolution written to: {solution_path}")

    return result


def _status_name(status):
    """Convert Gurobi status code to name."""
    status_map = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.CUTOFF: "CUTOFF",
        GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
        GRB.NODE_LIMIT: "NODE_LIMIT",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.NUMERIC: "NUMERIC",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
    }
    return status_map.get(status, f"UNKNOWN({status})")


# ============================================================================
# CLI Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="TDTSP Branch-and-Cut solver using Gurobi. "
                    "Based on Cordeau, Ghiani, Guerriero (2014).")
    parser.add_argument(
        "--instance_path", type=str, required=True,
        help="Path to the TDTSP instance JSON file.")
    parser.add_argument(
        "--solution_path", type=str, default="gurobi_solution_1.json",
        help="Output path for the solution JSON (default: "
             "gurobi_solution_1.json).")
    parser.add_argument(
        "--time_limit", type=int, default=3600,
        help="Gurobi time limit in seconds (default: 3600).")

    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    result = solve_tdtsp(args.instance_path, args.solution_path,
                         args.time_limit)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Status: {result['status_name']}")
    print(f"Objective value: {result['objective_value']:.6f}")
    print(f"Solve time: {result['solve_time']:.2f}s")
    print(f"Tour: {result['tour']}")
    print("=" * 70)


if __name__ == "__main__":
    main()
