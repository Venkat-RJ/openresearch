"""
Bi-Objective Single-Source Capacitated Facility Location Problem (BO-SSCFLP)
solved via epsilon-constraint method using Gurobi.

Reference: Boland, Charkhgard, Savelsbergh (2019)

NOTE: The BO-SSCFLP formulation is **NOT SPECIFIED IN THE PAPER** (deferred to
an online supplement). The formulation below is inferred from standard SSCFLP
literature.

Formulation
-----------
Sets:
    I: facilities  (i = 0..num_facilities-1)
    J: customers   (j = 0..num_customers-1)

Parameters (from JSON instance):
    f1[i], f2[i]        : fixed costs for opening facility i (obj1, obj2)
    c1[i][j], c2[i][j]  : assignment cost of customer j to facility i (obj1, obj2)
    d[j]                 : demand of customer j
    s[i]                 : capacity of facility i

Decision Variables:
    y[i]   in {0,1}  : 1 if facility i is opened
    x[i,j] in {0,1}  : 1 if customer j is assigned to facility i

Objectives (both minimize):
    z1 = sum_i f1[i]*y[i] + sum_i sum_j c1[i][j]*x[i,j]
    z2 = sum_i f2[i]*y[i] + sum_i sum_j c2[i][j]*x[i,j]

Constraints:
    sum_i x[i,j] = 1              for all j   (single-source assignment)
    sum_j d[j]*x[i,j] <= s[i]*y[i] for all i   (capacity)
    x[i,j] <= y[i]                for all i,j  (linking -- inferred assumption)
    x[i,j], y[i] in {0,1}

Algorithm: Epsilon-Constraint Method
-------------------------------------
1. Lexicographic min obj1 then obj2  -> upper-left point (z1_ul, z2_ul)
2. Lexicographic min obj2 then obj1  -> lower-right point (z1_lr, z2_lr)
3. Sweep eps from z2_ul - 1 down to z2_lr, constrain z2 <= eps, minimise z1.
   Each new non-dominated point updates eps = z2_new - 1.
4. Collect all non-dominated outcome vectors.

Since objectives are integer-valued (Z^2 outcome space), step by 1.
"""

import argparse
import json
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


def load_instance(path):
    with open(path, "r") as f:
        return json.load(f)


def build_model(data):
    """Build the base SSCFLP model (without objective or epsilon constraint)."""
    nI = data["num_facilities"]
    nJ = data["num_customers"]

    f1 = data["fixed_costs_obj1"]
    f2 = data["fixed_costs_obj2"]
    c1 = data["assignment_costs_obj1"]
    c2 = data["assignment_costs_obj2"]
    d = data["demands"]
    s = data["capacities"]

    m = gp.Model("BO_SSCFLP")
    m.setParam("Threads", 1)
    m.Params.OutputFlag = 0

    # Decision variables
    y = m.addVars(nI, vtype=GRB.BINARY, name="y")
    x = m.addVars(nI, nJ, vtype=GRB.BINARY, name="x")

    # Single-source assignment: each customer assigned to exactly one facility
    for j in range(nJ):
        m.addConstr(gp.quicksum(x[i, j] for i in range(nI)) == 1,
                    name=f"assign_{j}")

    # Capacity constraints
    for i in range(nI):
        m.addConstr(gp.quicksum(d[j] * x[i, j] for j in range(nJ)) <= s[i] * y[i],
                    name=f"cap_{i}")

    # Linking constraints: x[i,j] <= y[i]
    # NOTE: These linking constraints are an **inferred assumption**, not
    # explicitly stated in the paper (which defers formulation to online supplement).
    # They are standard in SSCFLP formulations and are implied by the capacity
    # constraints when demands are positive, but we include them explicitly
    # to strengthen the LP relaxation.
    for i in range(nI):
        for j in range(nJ):
            m.addConstr(x[i, j] <= y[i], name=f"link_{i}_{j}")

    # Build objective expressions
    z1_expr = (gp.quicksum(f1[i] * y[i] for i in range(nI))
               + gp.quicksum(c1[i][j] * x[i, j] for i in range(nI) for j in range(nJ)))
    z2_expr = (gp.quicksum(f2[i] * y[i] for i in range(nI))
               + gp.quicksum(c2[i][j] * x[i, j] for i in range(nI) for j in range(nJ)))

    m.update()
    return m, y, x, z1_expr, z2_expr


def eval_objectives(y, x, data):
    """Evaluate both objective values from the current solution."""
    nI = data["num_facilities"]
    nJ = data["num_customers"]
    f1 = data["fixed_costs_obj1"]
    f2 = data["fixed_costs_obj2"]
    c1 = data["assignment_costs_obj1"]
    c2 = data["assignment_costs_obj2"]

    z1 = sum(f1[i] * round(y[i].X) for i in range(nI))
    z1 += sum(c1[i][j] * round(x[i, j].X) for i in range(nI) for j in range(nJ))

    z2 = sum(f2[i] * round(y[i].X) for i in range(nI))
    z2 += sum(c2[i][j] * round(x[i, j].X) for i in range(nI) for j in range(nJ))

    return int(round(z1)), int(round(z2))


def extract_solution(y, x, data):
    """Extract the current solution as a dict."""
    nI = data["num_facilities"]
    nJ = data["num_customers"]
    open_facilities = [i for i in range(nI) if round(y[i].X) > 0.5]
    assignments = {}
    for j in range(nJ):
        for i in range(nI):
            if round(x[i, j].X) > 0.5:
                assignments[j] = i
                break
    return {"open_facilities": open_facilities, "assignments": assignments}


def solve_epsilon_constraint(data, time_limit):
    """
    Enumerate all non-dominated points of the BO-SSCFLP using the
    epsilon-constraint method.
    """
    start_time = time.time()
    pareto_front = []
    solutions = []

    def remaining_time():
        return max(0, time_limit - (time.time() - start_time))

    def time_expired():
        return time.time() - start_time >= time_limit

    # ------------------------------------------------------------------
    # Step 1: Lexicographic optimum  min z1, then min z2
    # ------------------------------------------------------------------
    m, y, x, z1_expr, z2_expr = build_model(data)
    m.Params.TimeLimit = remaining_time()
    m.setObjective(z1_expr, GRB.MINIMIZE)
    m.optimize()

    if m.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT) or m.SolCount == 0:
        # No feasible solution found at all
        return pareto_front, solutions

    z1_star = round(m.ObjVal)

    # Fix z1 = z1_star, minimize z2
    eps_constr_z1 = m.addConstr(z1_expr <= z1_star, name="fix_z1")
    m.setObjective(z2_expr, GRB.MINIMIZE)
    m.Params.TimeLimit = remaining_time()
    m.optimize()

    if m.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT) or m.SolCount == 0:
        # Return with just the first solution
        m.remove(eps_constr_z1)
        m.setObjective(z1_expr, GRB.MINIMIZE)
        m.Params.TimeLimit = remaining_time()
        m.optimize()
        if m.SolCount > 0:
            z1_val, z2_val = eval_objectives(y, x, data)
            pareto_front.append([z1_val, z2_val])
            solutions.append(extract_solution(y, x, data))
        return pareto_front, solutions

    z1_ul, z2_ul = eval_objectives(y, x, data)
    pareto_front.append([z1_ul, z2_ul])
    solutions.append(extract_solution(y, x, data))

    m.remove(eps_constr_z1)
    m.update()

    if time_expired():
        return pareto_front, solutions

    # ------------------------------------------------------------------
    # Step 2: Lexicographic optimum  min z2, then min z1
    # ------------------------------------------------------------------
    m.setObjective(z2_expr, GRB.MINIMIZE)
    m.Params.TimeLimit = remaining_time()
    m.optimize()

    if m.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT) or m.SolCount == 0:
        return pareto_front, solutions

    z2_star = round(m.ObjVal)

    eps_constr_z2 = m.addConstr(z2_expr <= z2_star, name="fix_z2")
    m.setObjective(z1_expr, GRB.MINIMIZE)
    m.Params.TimeLimit = remaining_time()
    m.optimize()

    if m.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT) or m.SolCount == 0:
        return pareto_front, solutions

    z1_lr, z2_lr = eval_objectives(y, x, data)

    # Only add if it is a genuinely different point
    if [z1_lr, z2_lr] != pareto_front[-1]:
        pareto_front.append([z1_lr, z2_lr])
        solutions.append(extract_solution(y, x, data))

    m.remove(eps_constr_z2)
    m.update()

    if time_expired():
        return pareto_front, solutions

    # ------------------------------------------------------------------
    # Step 3: Sweep epsilon from z2_ul - 1 down to z2_lr
    # ------------------------------------------------------------------
    # We already have the two extreme points.  Now find interior points.
    eps = z2_ul - 1

    # Add an epsilon constraint on z2 that we will update each iteration
    eps_constr = m.addConstr(z2_expr <= eps, name="eps_z2")
    m.setObjective(z1_expr, GRB.MINIMIZE)

    while eps >= z2_lr and not time_expired():
        # Update the epsilon constraint RHS
        eps_constr.RHS = eps

        m.Params.TimeLimit = remaining_time()
        m.optimize()

        if m.Status not in (GRB.OPTIMAL, GRB.TIME_LIMIT) or m.SolCount == 0:
            # No feasible solution with z2 <= eps; we are done
            break

        if m.Status == GRB.TIME_LIMIT and m.SolCount == 0:
            break

        z1_new, z2_new = eval_objectives(y, x, data)

        # Record this non-dominated point (skip if already recorded)
        if [z1_new, z2_new] not in pareto_front:
            pareto_front.append([z1_new, z2_new])
            solutions.append(extract_solution(y, x, data))

        # Next epsilon: just below the z2 value we found
        eps = z2_new - 1

    # Sort Pareto front by z1 (ascending), z2 (descending) for clean output
    combined = list(zip(pareto_front, solutions))
    combined.sort(key=lambda p: (p[0][0], -p[0][1]))
    pareto_front = [p for p, _ in combined]
    solutions = [s for _, s in combined]

    return pareto_front, solutions


def main():
    parser = argparse.ArgumentParser(
        description="Solve BO-SSCFLP via epsilon-constraint method (Gurobi)")
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to the JSON instance file")
    parser.add_argument("--solution_path", type=str,
                        default="gurobi_solution_1.json",
                        help="Path to write the JSON solution file")
    parser.add_argument("--time_limit", type=int, default=3600,
                        help="Total time limit in seconds across all solves")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    data = load_instance(args.instance_path)
    start_time = time.time()

    pareto_front, solutions = solve_epsilon_constraint(data, args.time_limit)

    elapsed = time.time() - start_time

    # Compute the best weighted-sum objective (lambda = 0.5) across all
    # non-dominated points, consistent with the paper's scalarisation.
    if pareto_front:
        best_ws = min(0.5 * z1 + 0.5 * z2 for z1, z2 in pareto_front)
        best_idx = next(
            idx for idx, (z1, z2) in enumerate(pareto_front)
            if 0.5 * z1 + 0.5 * z2 == best_ws
        )
    else:
        best_ws = None
        best_idx = None

    result = {
        "objective_value": best_ws,
        "num_nondominated_points": len(pareto_front),
        "pareto_front": pareto_front,
        "solutions": [
            {
                "z1": pf[0],
                "z2": pf[1],
                "open_facilities": sol["open_facilities"],
                "assignments": {str(k): v for k, v in sol["assignments"].items()},
            }
            for pf, sol in zip(pareto_front, solutions)
        ],
        "best_weighted_sum_index": best_idx,
        "elapsed_time": elapsed,
    }

    with open(args.solution_path, "w") as f:
        result["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(result, f, indent=2)

    print(f"Found {len(pareto_front)} non-dominated points in {elapsed:.2f}s")
    print(f"Best weighted-sum (0.5*z1 + 0.5*z2) = {best_ws}")
    print(f"Pareto front: {pareto_front}")
    print(f"Solution written to {args.solution_path}")


if __name__ == "__main__":
    main()
