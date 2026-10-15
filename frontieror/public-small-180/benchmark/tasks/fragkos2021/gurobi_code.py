"""
Gurobi implementation of the Stochastic Reliable Uncapacitated Facility Location
Problem (RUFLP) via the Cutting Plane Algorithm.

Reference:
    Li, Li, Shu, Song, Zhang (2021) "A General Model and Efficient Algorithms for
    Reliable Facility Location Problem under Uncertain Disruptions."
    INFORMS Journal on Computing 34(1):407-426.

Model:
    min_{x in {0,1}^|J|}  sum_{j in J} f_j x_j + E_{p^dagger}[Q(x, tilde_xi)]

    where Q(x, xi) = sum_{i in I} d_i * min_{j in J union {0} | x_j * xi_j = 1} c_{ij}

    For the stochastic RUFLP (instance model_type = "stochastic_RUFLP"), P = {p^dagger}
    is a singleton, so the max over P collapses to a single expectation.

Algorithm:
    Cutting Plane (CP) algorithm (Algorithm CP in paper, Section 4).
    The relaxed master problem (RP) is a MILP solved by Gurobi using LazyConstraints
    callbacks. For each integer solution found, the separation problem is evaluated via
    Algorithm 1 (O(|I||J|log|J|)), and a valid Benders-type cut is added via Algorithm 2
    (O(|I||J|^2)) if needed.

    Termination: relative optimality gap <= 0.5% or time limit reached.
"""

import argparse
import json
import os
import tempfile
import time
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


# ---------------------------------------------------------------------------
# Algorithm 1: Evaluate E_{p^dagger}[Q(x*, tilde_xi)]
# ---------------------------------------------------------------------------

def algorithm1_eval(x_vals, q, c, penalty, demands, n_fac, n_cus):
    """
    Algorithm 1 (Paper Section 4.1): Evaluate E_{p^dagger}[Q(x*, tilde_xi)]
    for independent facility disruptions.

    For independent disruptions:
        Prob(xi_{j_t}=0 | xi_{j_1}=0, ..., xi_{j_{t-1}}=0) = 1 - q_{j_t}  (O(1))

    Variable gamma = Prob(j_1, ..., j_{t-1} all disrupted).

    Lines 7-10 from Algorithm 1 (PDF p.16):
        cost += gamma * (1 - Prob(xi_{j_t}=0 | all prev disrupted)) * d_i * c_{i,j_t}
             = gamma * q_{j_t} * d_i * c_{i,j_t}          [since q_{j_t} = survival prob]
        gamma *= Prob(xi_{j_t}=0 | all prev disrupted) = (1 - q_{j_t})

    Dummy facility (index = n_fac): always online (xi_0 = 1, so q_0 = 1, Prob(disrupt)=0).
    Its cost is penalty[i] = c_{i0}.

    Complexity: O(|I||J| log |J|)

    Args:
        x_vals  : list[float], facility open/close decisions (> 0.5 treated as open)
        q       : list[float], q[j] = survival probability of facility j
        c       : list[list[float]], c[i][j] = unit shipment cost from j to i
        penalty : list[float], penalty[i] = c_{i0} (penalty cost for customer i)
        demands : list[float], demands[i] = d_i
        n_fac   : int, number of facilities |J|
        n_cus   : int, number of customers |I|

    Returns:
        float: E_{p^dagger}[Q(x*, tilde_xi)]
    """
    DUMMY = n_fac  # dummy facility index (represents the penalty option)
    open_facs = [j for j in range(n_fac) if x_vals[j] > 0.5]

    total = 0.0
    for i in range(n_cus):
        # Step A (Lines 1-3): sort open facilities by c[i][j] ascending; dummy last
        # (dummy always last because penalty[i] >= c[i][j] for all j by assumption)
        L_i = sorted(open_facs, key=lambda j: c[i][j]) + [DUMMY]

        # Step B (Lines 4-11): accumulate expected cost for customer i
        gamma = 1.0  # P(j_1,...,j_{t-1} all disrupted)
        for j_t in L_i:
            if j_t == DUMMY:
                q_t = 1.0       # dummy never disrupted
                c_t = penalty[i]
            else:
                q_t = q[j_t]    # survival probability P(xi_{j_t}=1)
                c_t = c[i][j_t]

            # Prob(j_t online | all prev disrupted) = q_t   (by independence)
            # Prob(j_t disrupted | all prev disrupted) = 1 - q_t
            total += gamma * q_t * demands[i] * c_t
            gamma *= (1.0 - q_t)  # update joint disruption probability

    return total


# ---------------------------------------------------------------------------
# Algorithm 2: Formulate valid cut (Equation 5) for the stochastic case
# ---------------------------------------------------------------------------

def algorithm2_cut(x_vals, q, c, penalty, demands, n_fac, n_cus):
    """
    Algorithm 2 (Paper Section 4.1, PDF p.17): Formulate the valid cut (Eq. 5)
    for given x* and p^dagger (stochastic independent disruptions).

    Cut:  eta >= cost + sum_{j in J | x_j^*=0} (cost_j[j] - cost) * x_j

    where:
        cost     = E_{p^dagger}[Q(x^*, tilde_xi)]
        cost_j[j] = E_{p^dagger}[Q(x^* v z_j, tilde_xi)]  for closed facility j

    Note: cost_j[j] <= cost for all j (adding a facility cannot increase expected cost),
    so (cost_j[j] - cost) <= 0. These are valid supporting hyperplane coefficients.

    Complexity: O(|I||J|^2)
    """
    DUMMY = n_fac
    open_facs = [j for j in range(n_fac) if x_vals[j] > 0.5]
    closed_facs = [j for j in range(n_fac) if x_vals[j] <= 0.5]

    # Line 1: Step A — pre-compute sorted L_i for all customers
    L_base = []
    for i in range(n_cus):
        L_base.append(sorted(open_facs, key=lambda jj: c[i][jj]) + [DUMMY])

    # Lines 3-6 (Step B of Alg 1): compute cost = E[Q(x*)]
    cost = 0.0
    for i in range(n_cus):
        gamma = 1.0
        for j_t in L_base[i]:
            q_t = 1.0 if j_t == DUMMY else q[j_t]
            c_t = penalty[i] if j_t == DUMMY else c[i][j_t]
            cost += gamma * q_t * demands[i] * c_t
            gamma *= (1.0 - q_t)

    # Lines 7-17 (Step C): for each closed facility j, compute cost_j = E[Q(x^* v z_j)]
    cost_j = {}
    for j in closed_facs:
        val = 0.0
        for i in range(n_cus):
            # Lines 10-15: insert j into L_base[i] at its sorted position (before dummy)
            L_i = L_base[i]
            pos = 0
            # Advance past facilities with c[i][*] <= c[i][j] (do not go past dummy)
            while pos < len(L_i) - 1 and c[i][L_i[pos]] <= c[i][j]:
                pos += 1
            L_new = L_i[:pos] + [j] + L_i[pos:]

            # Step C (Lines 11-14): same accumulation as Algorithm 1 Step B
            gamma = 1.0
            for j_t in L_new:
                q_t = 1.0 if j_t == DUMMY else q[j_t]
                c_t = penalty[i] if j_t == DUMMY else c[i][j_t]
                val += gamma * q_t * demands[i] * c_t
                gamma *= (1.0 - q_t)
            # Line 15: remove j from L_i (L_new is local; L_base[i] is unchanged)

        cost_j[j] = val

    return cost, cost_j


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Gurobi-based Cutting Plane solver for Stochastic RUFLP"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON problem instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path for the output solution JSON file")
    parser.add_argument("--time_limit", type=int, default=3600,
                        help="Maximum solver runtime in seconds (default: 3600)")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    # ------------------------------------------------------------------
    # Load problem instance
    # ------------------------------------------------------------------
    with open(args.instance_path, "r") as f:
        instance = json.load(f)

    n_fac = instance["num_facilities"]
    n_cus = instance["num_customers"]
    nodes = instance["nodes"]

    demands = [nodes[i]["demand"] for i in range(n_cus)]
    f_costs = [nodes[j]["fixed_setup_cost"] for j in range(n_fac)]
    # q[j] = survival probability = P(facility j is online)
    q = [nodes[j]["survival_probability"] for j in range(n_fac)]
    # c[i][j] = unit shipment cost from facility j to customer i
    c = instance["transportation_cost_matrix"]
    # penalty[i] = c_{i0}: unit penalty cost when customer i has no open facility
    penalty = instance["penalty_cost"]

    start_time = time.time()

    # ------------------------------------------------------------------
    # Atomic JSON writer. Used both for the up-front stub and for the
    # incremental flush on every new incumbent inside the callback, so that
    # if the wrapper SIGTERMs the process, the on-disk file always reflects
    # the latest captured incumbent (or the trivial stub if Gurobi found
    # nothing yet) — never a half-written or stale-dimension file.
    # ------------------------------------------------------------------
    def _atomic_write_solution(path, x_vec, ub, elapsed_s, cuts, solver_result=None):
        payload = {
            "objective_value": ub,
            "facility_locations": x_vec,
            "instance_id": instance.get("instance_id", ""),
            "total_time_seconds": round(elapsed_s, 4),
            "cuts_added": cuts,
        }
        if solver_result is not None:
            payload.update(solver_result)
        dirn = _os.path.dirname(_os.path.abspath(path)) or "."
        fd, tmp = tempfile.mkstemp(prefix=_os.path.basename(path) + ".",
                                   suffix=".tmp", dir=dirn)
        try:
            with _os.fdopen(fd, "w") as f:
                payload["runtime"] = time.time() - _GUROBI_CODE_START_TIME
                json.dump(payload, f, indent=2)
            _os.replace(tmp, path)
        except Exception:
            try:
                _os.unlink(tmp)
            except OSError:
                pass
            raise

    # ------------------------------------------------------------------
    # Preemptive trivial-fallback solution (open no facilities, pay all penalties).
    # Writes a correctly-sized stub up front so that if the solver is killed by
    # SIGTERM at the wall-time limit, a valid-dimension solution is still on disk.
    # The callback overwrites this with real incumbents as they are found.
    # ------------------------------------------------------------------
    trivial_x = [0] * n_fac
    trivial_obj = sum(demands[i] * penalty[i] for i in range(n_cus))
    _atomic_write_solution(args.solution_path, trivial_x, trivial_obj, 0.0, 0)

    # ------------------------------------------------------------------
    # State shared between callback and main thread
    # ------------------------------------------------------------------
    best = {
        "ub": float("inf"),
        "x": None,
        "cuts": 0,
    }

    # ------------------------------------------------------------------
    # Build Gurobi relaxed master problem (RP)
    # ------------------------------------------------------------------
    model = gp.Model("RUFLP_Stochastic_Gurobi")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", args.time_limit)
    model.setParam("LazyConstraints", 1)
    model.setParam("OutputFlag", 0)

    # Binary facility-open decisions x_j
    x = model.addVars(n_fac, vtype=GRB.BINARY, name="x")

    # eta >= 0: epigraph variable bounding worst-case expected cost.
    # INFERRED ASSUMPTION: lb=0 is valid because E[Q(x,xi)] >= 0 for all x
    # (all transportation costs are non-negative). The paper starts with an empty
    # Lambda set (no cuts), which would leave eta unbounded; setting lb=0 provides
    # the minimal valid tightening without adding explicit initial cuts.
    eta = model.addVar(lb=0.0, name="eta")

    # Objective: min sum_j f_j x_j + eta  (Equation 4)
    model.setObjective(
        gp.quicksum(f_costs[j] * x[j] for j in range(n_fac)) + eta,
        GRB.MINIMIZE,
    )

    # ------------------------------------------------------------------
    # LazyConstraints callback — implements Steps 2 and 3 of Algorithm CP
    # ------------------------------------------------------------------
    def separation_callback(model, where):
        if where != GRB.Callback.MIPSOL:
            return

        # Check elapsed time; terminate if over budget
        if time.time() - start_time >= args.time_limit:
            model.terminate()
            return

        # Retrieve current integer-feasible RP solution
        x_vals = [model.cbGetSolution(x[j]) for j in range(n_fac)]
        eta_val = model.cbGetSolution(eta)

        # Step 2: Algorithm 1 — evaluate E_{p^dagger}[Q(x*, tilde_xi)]
        e_cost = algorithm1_eval(x_vals, q, c, penalty, demands, n_fac, n_cus)

        # Track best upper bound: UB = setup cost + expected transportation cost
        setup = sum(f_costs[j] * round(x_vals[j]) for j in range(n_fac))
        ub = setup + e_cost
        if ub < best["ub"]:
            best["ub"] = ub
            best["x"] = [round(x_vals[j]) for j in range(n_fac)]
            # Flush the new incumbent to disk atomically. If the wrapper
            # SIGTERMs us before model.optimize() returns, this file is what
            # the feasibility checker will read.
            _atomic_write_solution(
                args.solution_path,
                best["x"],
                best["ub"],
                time.time() - start_time,
                best["cuts"],
            )

        # Step 3: If eta underestimates E[Q], separation violated — add valid cut
        if eta_val < e_cost - 1e-6:
            # Algorithm 2: compute cut coefficients
            cost, cost_j = algorithm2_cut(x_vals, q, c, penalty, demands, n_fac, n_cus)

            # Valid cut (Equation 5):
            #   eta >= cost + sum_{j: x_j^*=0} (cost_j[j] - cost) * x_j
            # Coefficients (cost_j[j] - cost) <= 0: opening more facilities reduces cost.
            rhs = cost + gp.quicksum(
                (cost_j[j] - cost) * x[j] for j in cost_j
            )
            model.cbLazy(eta >= rhs)
            best["cuts"] += 1

    # ------------------------------------------------------------------
    # Solve
    # ------------------------------------------------------------------
    model.optimize(separation_callback)

    # ------------------------------------------------------------------
    # Extract best solution found
    # ------------------------------------------------------------------
    # If no solution was captured in the callback, try reading final model solution
    if best["x"] is None and model.SolCount > 0:
        x_vals = [x[j].X for j in range(n_fac)]
        e_cost = algorithm1_eval(x_vals, q, c, penalty, demands, n_fac, n_cus)
        setup = sum(f_costs[j] * round(x_vals[j]) for j in range(n_fac))
        best["ub"] = setup + e_cost
        best["x"] = [round(x_vals[j]) for j in range(n_fac)]

    # Fallback: trivial solution opens no facilities (all customers pay penalty).
    # INFERRED ASSUMPTION: if Gurobi found no feasible solution, use x=0.
    if best["x"] is None:
        best["x"] = [0] * n_fac
        best["ub"] = sum(demands[i] * penalty[i] for i in range(n_cus))

    total_time = time.time() - start_time

    # ------------------------------------------------------------------
    # Final write: covers the case where Gurobi exited cleanly within the
    # time limit, or pulled an incumbent from SolCount that the callback
    # never saw. The callback already kept the on-disk file fresh during
    # the solve, so this is a tail-end refresh, not the primary save.
    # ------------------------------------------------------------------
    _atomic_write_solution(
        args.solution_path,
        best["x"],
        best["ub"],
        total_time,
        best["cuts"],
        {
            "solver_status": model.Status,
            "best_bound": model.ObjBound if model.SolCount > 0 else None,
            "mip_gap": model.MIPGap if model.SolCount > 0 else None,
        },
    )

    print(f"Objective value : {best['ub']:.4f}")
    print(f"Cuts added      : {best['cuts']}")
    print(f"Total time      : {total_time:.2f}s")
    print(f"Solution saved  : {args.solution_path}")


if __name__ == "__main__":
    main()
