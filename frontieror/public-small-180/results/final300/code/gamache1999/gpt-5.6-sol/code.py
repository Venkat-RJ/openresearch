"""
Gurobi implementation of the Generalized Set Partitioning Problem
for Aircrew Rostering (Gamache, Soumis, Marquis, Desrosiers, 1999).

The model:
  Min  sum_{p=1}^{n} c_p * e_p                                    (1)
  s.t. sum_{k=1}^{m} sum_{r in R_k} a_{pr} * x_r + e_p = b_p     (2)
       sum_{r in R_k} x_r = 1                                      (3)
       e_p >= 0                                                     (4)
       x_r in {0, 1}                                                (5)

Since explicit column enumeration is needed for direct Gurobi solve,
we build employee-specific acyclic networks and enumerate feasible
source-to-sink paths subject to resource constraints.
"""

import argparse
import json
import time
import math
from collections import defaultdict
from itertools import count

import os,sys,random
_license=os.environ.get('GRB_LICENSE_FILE')
if _license and not os.path.exists(_license):
    os.environ.pop('GRB_LICENSE_FILE')
from solution_logger import SolutionLogger
import gurobipy as gp
_original_optimize=gp.Model.optimize
def _logged_optimize(self, callback=None):
    self.Params.Seed=0
    self.Params.MIPGap=1e-4
    self.Params.NumericFocus=0
    self.Params.Threads=1
    self.Params.TimeLimit=max(.1,self.Params.TimeLimit-4)
    logger=SolutionLogger(sys.argv[sys.argv.index('--log_path')+1],sense='minimize') if '--log_path' in sys.argv else None
    if logger:logger.min_interval=0;logger.start_time=_GUROBI_CODE_START_TIME
    def cb(model,where):
        if logger and where==gp.GRB.Callback.MIPSOL:
            logger.log(model.cbGet(gp.GRB.Callback.MIPSOL_OBJ))
        if callback:callback(model,where)
    return _original_optimize(self,cb)
gp.Model.optimize=_logged_optimize
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
# =============================================================================
# DATA LOADING
# =============================================================================

def load_instance(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


# =============================================================================
# NETWORK CONSTRUCTION
# =============================================================================

def build_network_and_enumerate(data, crew):
    """
    Build an acyclic network for one crew member and enumerate feasible
    source-to-sink paths (rosters) via resource-constrained DP.

    Returns list of dicts: [{pairings: set, unproductive_time: float, flight_hours: float}, ...]
    """
    deadline=time.time()+max(.05,data.get('_enumeration_budget',.5))
    horizon = data["planning_horizon_days"]  # 30 days
    work_rules = data["work_rules"]
    pairings = data["pairings"]

    pre_assigned_days = set(crew["pre_assigned_days"])
    qualifications = set(crew["qualifications"])
    flight_hours_prev = crew["flight_hours_previous_15_days"]

    # Determine which pairings this crew member can work
    eligible_pairings = []
    for p in pairings:
        # Check aircraft type qualification
        if p["aircraft_type"] not in qualifications:
            continue
        # Check overlap with pre-assigned days
        pairing_days = set(range(p["start_day"], p["start_day"] + p["duration_days"]))
        if pairing_days & pre_assigned_days:
            continue
        eligible_pairings.append(p)

    # Build time events (nodes)
    # Nodes represent time points. We use (day, hour) tuples.
    # Source = start of month, Sink = end of month
    SOURCE = "SOURCE"
    SINK = "SINK"

    # For simplicity, we model at day granularity with hour info on arcs.
    # Nodes are day boundaries: day 0 start, day 1 start, ..., day horizon start (= end of month)
    # Plus pairing-specific start/end nodes.

    # We use a DP approach directly on a timeline.
    # State: (current_day_position, resource_vector)
    # Activities are placed as intervals on the timeline.

    # For tractability with small instances, we enumerate rosters by
    # trying all subsets of eligible pairings that don't overlap each other
    # or pre-assigned days, and that satisfy resource constraints.
    # Given n=40 pairings and ~20 eligible per crew member, we use
    # a DFS/backtracking approach.

    # Sort eligible pairings by start_day, then by departure_hour
    eligible_pairings.sort(key=lambda p: (p["start_day"], p["departure_hour"]))

    # Pre-assigned activities as intervals
    pre_assigned_intervals = []
    for act in crew["pre_assigned_activities"]:
        pre_assigned_intervals.append((act["start_day"], act["end_day"]))

    # Check if day d is a pre-assigned day
    def is_preassigned(d):
        return d in pre_assigned_days

    # Total minutes in the planning horizon
    total_minutes = horizon * 24 * 60

    # Week boundaries: Monday=day 0, week = days [7*w, 7*w+6]
    # **INFERRED ASSUMPTION**: Day 0 of the planning horizon is a Monday.
    # The paper says "week beginning Monday at 0:00 hours and ending Sunday at 24:00 hours."
    # NOT SPECIFIED IN PAPER: exact day-of-week alignment for the test instance.
    # We assume day 0 = Monday for simplicity.
    def get_week(day):
        return day // 7

    num_weeks = math.ceil(horizon / 7)

    rosters = []

    def pairing_interval(p):
        """Return (start_day, end_day_exclusive) for a pairing."""
        return (p["start_day"], p["start_day"] + p["duration_days"])

    def intervals_overlap(a_start, a_end, b_start, b_end):
        """Check if [a_start, a_end) overlaps [b_start, b_end)."""
        return a_start < b_end and b_start < a_end

    def check_resources(selected_pairings):
        """
        Check all 6 resource constraints for a given set of selected pairings.
        Returns (feasible, unproductive_time_minutes) or (False, None).
        """
        # Build timeline of occupied days
        occupied_days = set(pre_assigned_days)
        total_flight_hours = 0.0
        pairing_ids = set()
        has_cross_pairing = False

        pairing_events = []  # (start_day, end_day_exclusive, pairing)
        for p in selected_pairings:
            s = p["start_day"]
            e = s + p["duration_days"]
            pairing_events.append((s, e, p))
            for d in range(s, e):
                occupied_days.add(d)
            total_flight_hours += p["flight_hours"]
            pairing_ids.add(p["id"])
            if p["is_cross_pairing"]:
                if has_cross_pairing:
                    return False, None  # Resource 6: at most 1 cross-pairing
                has_cross_pairing = True

        # Resource 4: Flight time limitations
        # 85 hours limit for rolling period (day -15 to day 14, i.e., first 15 days)
        # 77 hours limit for the whole month
        flight_hours_first_15 = sum(
            p["flight_hours"] for p in selected_pairings
            if p["start_day"] < 15
        )
        if flight_hours_first_15 + flight_hours_prev > 85:
            return False, None
        if total_flight_hours > 77:
            return False, None

        # Resource 5: Departures before 8:00 a.m.
        # At most 2 consecutive days with departure before 8am
        early_departure_days = set()
        for p in selected_pairings:
            if p["departure_hour"] < 8.0:
                early_departure_days.add(p["start_day"])
        # Check for 3 consecutive days
        for d in early_departure_days:
            if (d + 1) in early_departure_days and (d + 2) in early_departure_days:
                return False, None

        # Resource 1 & 2: Weekly and monthly rest periods
        # Need a 48-hour rest each week, and one monthly rest (additional 48h on a weekly rest)
        # A periodic break is a gap of >= 48 hours (for weekly) or >= 96 hours (for monthly)

        # Build sorted list of all activity intervals
        all_intervals = []
        for p in selected_pairings:
            s = p["start_day"]
            e = s + p["duration_days"]
            # Convert to hours for more precision
            start_h = s * 24 + p["departure_hour"]
            end_h = (s + p["duration_days"] - 1) * 24 + p["return_hour"]
            # If return_hour < departure_hour on last day, pairing wraps
            if p["duration_days"] == 1:
                end_h = s * 24 + p["return_hour"]
                if end_h <= start_h:
                    end_h = start_h + p["duration_minutes"] / 60.0
            all_intervals.append((start_h, end_h, "pairing", p["id"]))

        for act in crew["pre_assigned_activities"]:
            start_h = act["start_day"] * 24
            end_h = (act["end_day"] + 1) * 24  # end_day is inclusive
            all_intervals.append((start_h, end_h, "preassign", -1))

        all_intervals.sort(key=lambda x: x[0])

        # Merge overlapping intervals to find gaps
        if not all_intervals:
            # No activities at all - entire month is free
            # Weekly and monthly rest easily satisfied
            unproductive = total_minutes
            return True, unproductive

        merged = []
        for intv in all_intervals:
            if merged and intv[0] <= merged[-1][1]:
                merged[-1] = (merged[-1][0], max(merged[-1][1], intv[1]))
            else:
                merged.append([intv[0], intv[1]])

        # Gaps between merged intervals (and at start/end of month)
        gaps = []
        month_start_h = 0
        month_end_h = horizon * 24

        if merged[0][0] > month_start_h:
            gaps.append((month_start_h, merged[0][0]))
        for i in range(len(merged) - 1):
            gap_start = merged[i][1]
            gap_end = merged[i + 1][0]
            if gap_end > gap_start:
                gaps.append((gap_start, gap_end))
        if merged[-1][1] < month_end_h:
            gaps.append((merged[-1][1], month_end_h))

        # Resource 3: Spread between breaks
        # No more than 8 days (192 hours) between two periodic breaks (>= 48h gaps)
        # or between a periodic break and annual leave, etc.
        # **INFERRED ASSUMPTION**: A gap >= 48 hours counts as a periodic break.
        # Reserve blocks in pre-assignments may extend the limit to 9 days.
        periodic_breaks = []
        for g_start, g_end in gaps:
            duration = g_end - g_start
            if duration >= 48:
                periodic_breaks.append((g_start, g_end))

        # Also count pre-assigned annual leave as break-like
        for act in crew["pre_assigned_activities"]:
            if act["type"] in ("annual_leave",):
                act_start = act["start_day"] * 24
                act_end = (act["end_day"] + 1) * 24
                periodic_breaks.append((act_start, act_end))

        periodic_breaks.sort()
        # Remove duplicates / merge
        if periodic_breaks:
            merged_breaks = [list(periodic_breaks[0])]
            for b in periodic_breaks[1:]:
                if b[0] <= merged_breaks[-1][1]:
                    merged_breaks[-1][1] = max(merged_breaks[-1][1], b[1])
                else:
                    merged_breaks.append(list(b))
            periodic_breaks = merged_breaks

        # Check spread: from start of month to first break, between breaks, last break to end
        check_points = [month_start_h]
        for b_start, b_end in periodic_breaks:
            check_points.append(b_start)
            check_points.append(b_end)
        check_points.append(month_end_h)

        max_spread_hours = work_rules["max_spread_between_breaks_days"] * 24  # 192 hours

        for i in range(0, len(check_points) - 1, 2):
            spread = check_points[i + 1] - check_points[i] if i + 1 < len(check_points) else 0
            # **INFERRED ASSUMPTION**: The spread check is between end of one break and start of next
            pass  # Relaxed check for small instances; strict check below

        # More precise spread check:
        if periodic_breaks:
            # From month start to first break start
            if periodic_breaks[0][0] - month_start_h > max_spread_hours:
                return False, None
            # Between consecutive breaks
            for i in range(len(periodic_breaks) - 1):
                if periodic_breaks[i + 1][0] - periodic_breaks[i][1] > max_spread_hours:
                    return False, None
            # From last break end to month end
            if month_end_h - periodic_breaks[-1][1] > max_spread_hours:
                return False, None
        else:
            # No periodic breaks at all - check if entire month fits in 8 days
            if month_end_h - month_start_h > max_spread_hours:
                return False, None

        # Resource 1: Weekly rest - check each week has a 48h rest
        # **INFERRED ASSUMPTION**: We check that within or overlapping each week
        # (Mon-Sun), there is a gap >= 48 hours. Weekly rest may overflow up to
        # 30 hours into the following week.
        weekly_rest_ok = True
        for w in range(num_weeks):
            week_start = w * 7 * 24
            week_end = min((w + 1) * 7 * 24, month_end_h)
            # Allow overflow: rest can start up to 30h before week_end
            # and extend into next week. Rest cannot start after Sunday 6am.
            # Sunday 6am = week_start + 6*24 + 6 = week_start + 150
            sunday_6am = week_start + 6 * 24 + 6

            found_rest = False
            for g_start, g_end in gaps:
                # Gap overlaps this week
                gap_in_week_start = max(g_start, week_start)
                gap_in_week_end = min(g_end, week_end + 30)  # allow 30h overflow
                duration_in_context = g_end - g_start
                if duration_in_context >= 48:
                    # Check if this rest is relevant to this week
                    if g_start < week_end and g_end > week_start:
                        # Rest must not begin on Sunday between 6am and midnight
                        if not (sunday_6am <= g_start < week_start + 7 * 24):
                            found_rest = True
                            break

            if not found_rest and week_end - week_start >= 48:
                # For partial last week, be lenient
                if w < num_weeks - 1 or (month_end_h - week_start) >= 4 * 24:
                    weekly_rest_ok = False
                    break

        if not weekly_rest_ok:
            return False, None

        # Resource 2: Monthly rest - at least one gap >= 96 hours
        # **INFERRED ASSUMPTION**: Monthly rest = one gap >= 96h (48h weekly + 48h additional)
        # The 48h additional may be reduced by 12h for every 7 inactive pre-assigned days.
        inactive_preassign_days = 0
        for act in crew["pre_assigned_activities"]:
            if act["type"] != "pairing":  # All pre-assigned types are inactive
                inactive_preassign_days += act["duration_days"]

        monthly_rest_reduction = (inactive_preassign_days // 7) * 12
        required_monthly_rest = max(48, 96 - monthly_rest_reduction)  # At least 48h (weekly rest)

        has_monthly_rest = False
        for g_start, g_end in gaps:
            if g_end - g_start >= required_monthly_rest:
                has_monthly_rest = True
                break

        if not has_monthly_rest:
            return False, None

        # Calculate unproductive time
        # Unproductive time = total month time - pairing time - pre-assigned activity time
        pairing_time = sum(p["duration_minutes"] for p in selected_pairings)
        preassign_time = sum(act["duration_days"] * 24 * 60 for act in crew["pre_assigned_activities"])
        unproductive = total_minutes - pairing_time - preassign_time
        if unproductive < 0:
            unproductive = 0

        return True, unproductive

    # Enumerate rosters via backtracking DFS
    # We try to select subsets of eligible_pairings that don't overlap
    def enumerate_rosters():
        """Generate feasible rosters by backtracking over eligible pairings."""
        n_eligible = len(eligible_pairings)

        # Always include the "empty roster" (no pairings assigned)
        feasible, unproductive = check_resources([])
        if feasible:
            rosters.append({
                "pairings": set(),
                "unproductive_time": unproductive,
                "flight_hours": 0.0
            })

        # DFS with pruning
        def dfs(idx, selected, occupied_days_set, flight_hours_accum):
            for i in range(idx, n_eligible):
                if time.time()>=deadline or len(rosters)>=6000:
                    return
                p = eligible_pairings[i]
                p_days = set(range(p["start_day"], p["start_day"] + p["duration_days"]))

                # Check overlap
                if p_days & occupied_days_set:
                    continue

                # Quick flight hour check
                new_fh = flight_hours_accum + p["flight_hours"]
                if new_fh > 77:
                    continue
                if p["start_day"] < 15:
                    # Rough check for rolling limit
                    fh_first15 = sum(
                        pp["flight_hours"] for pp in selected if pp["start_day"] < 15
                    ) + p["flight_hours"]
                    if fh_first15 + crew["flight_hours_previous_15_days"] > 85:
                        continue

                selected.append(p)
                new_occupied = occupied_days_set | p_days

                feasible, unproductive = check_resources(selected)
                if feasible:
                    rosters.append({
                        "pairings": set(pp["id"] for pp in selected),
                        "unproductive_time": unproductive,
                        "flight_hours": new_fh
                    })

                dfs(i + 1, selected, new_occupied, new_fh)

                selected.pop()

        dfs(0, [], set(pre_assigned_days), 0.0)

    enumerate_rosters()
    limit=max(1,1800//len(data['crew_members']))
    if len(rosters)>limit:
        selected=[]
        empty=next((r for r in rosters if not r['pairings']),None)
        if empty is not None:selected.append(empty)
        pool=[r for r in rosters if r is not empty]
        frequency=defaultdict(int)
        while pool and len(selected)<limit:
            chosen=max(pool,key=lambda r:sum(pairings[p]['duration_minutes']/(1+frequency[p]) for p in r['pairings']))
            selected.append(chosen);pool.remove(chosen)
            for p in chosen['pairings']:frequency[p]+=1
        rosters=selected
    return rosters


# =============================================================================
# GUROBI MODEL
# =============================================================================

def solve_model(data, all_rosters, time_limit):
    """
    Build and solve the generalized set partitioning problem using Gurobi.

    Min sum_{p} c_p * e_p
    s.t. sum_{k} sum_{r in R_k} a_{pr} * x_r + e_p = b_p   for all p
         sum_{r in R_k} x_r = 1                              for all k
         e_p >= 0
         x_r in {0, 1}
    """
    n_pairings = len(data["pairings"])
    n_crew = len(data["crew_members"])
    pairings = data["pairings"]

    model = gp.Model("AircrewRostering")
    model.setParam("Threads", 1)
    model.setParam("TimeLimit", time_limit)
    model.setParam("OutputFlag", 1)

    # Variables
    # x_r for each roster
    x_vars = {}
    for k in range(n_crew):
        for r_idx, roster in enumerate(all_rosters[k]):
            x_vars[(k, r_idx)] = model.addVar(
                vtype=GRB.BINARY,
                name=f"x_{k}_{r_idx}"
            )

    # e_p slack variables
    e_vars = {}
    for p_idx in range(n_pairings):
        e_vars[p_idx] = model.addVar(
            lb=0.0,
            vtype=GRB.CONTINUOUS,
            obj=pairings[p_idx]["duration_minutes"],  # c_p = duration in minutes
            name=f"e_{p_idx}"
        )

    model.update()

    # Objective: Min sum c_p * e_p (already set via obj parameter)
    model.setAttr("ModelSense", GRB.MINIMIZE)

    # Constraint (2): Pairing coverage
    for p_idx in range(n_pairings):
        bp = pairings[p_idx]["coverage_requirement"]
        expr = gp.LinExpr()
        for k in range(n_crew):
            for r_idx, roster in enumerate(all_rosters[k]):
                if p_idx in roster["pairings"]:
                    expr.addTerms(1.0, x_vars[(k, r_idx)])
        expr.addTerms(1.0, e_vars[p_idx])
        model.addConstr(expr == bp, name=f"cover_{p_idx}")

    # Constraint (3): One roster per crew member
    for k in range(n_crew):
        expr = gp.LinExpr()
        for r_idx in range(len(all_rosters[k])):
            expr.addTerms(1.0, x_vars[(k, r_idx)])
        model.addConstr(expr == 1, name=f"assign_{k}")

    model.update()
    model.optimize()

    # Extract solution
    solution = {
        "objective_value": None,
        "status": None,
        "rosters": {},
        "uncovered_pairings": [],
        "covered_pairings": []
    }

    if model.SolCount > 0:
        solution["objective_value"] = model.ObjVal
        solution["status"] = "optimal" if model.Status == GRB.OPTIMAL else "feasible"

        # Extract roster assignments
        for k in range(n_crew):
            for r_idx, roster in enumerate(all_rosters[k]):
                if x_vars[(k, r_idx)].X > 0.5:
                    solution["rosters"][str(k)] = {
                        "roster_index": r_idx,
                        "pairings": sorted(list(roster["pairings"])),
                        "unproductive_time": roster["unproductive_time"]
                    }
                    break

        # Extract uncovered pairings
        for p_idx in range(n_pairings):
            ep_val = e_vars[p_idx].X
            if ep_val > 0.5:
                solution["uncovered_pairings"].append({
                    "pairing_id": p_idx,
                    "uncovered_count": round(ep_val),
                    "duration_minutes": pairings[p_idx]["duration_minutes"]
                })
            else:
                solution["covered_pairings"].append(p_idx)

        total_uncovered_duration = sum(
            pairings[p_idx]["duration_minutes"] * round(e_vars[p_idx].X)
            for p_idx in range(n_pairings)
            if e_vars[p_idx].X > 0.5
        )
        solution["total_uncovered_duration_minutes"] = total_uncovered_duration
    else:
        solution["status"] = "infeasible"
        solution["objective_value"] = None

    return solution


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Gurobi solver for Aircrew Rostering (Gamache et al. 1999)"
    )
    parser.add_argument("--instance_path", type=str, required=True,
                        help="Path to JSON instance file")
    parser.add_argument("--solution_path", type=str, required=True,
                        help="Path for output solution JSON")
    parser.add_argument("--time_limit", type=int, required=True,
                        help="Maximum solver runtime in seconds")
    parser.add_argument("--log_path", type=str, default=None, help="Path to log incumbent solutions")
    args = parser.parse_args()
    install_gurobi_logger(args.log_path)

    print("=" * 60)
    print("Aircrew Rostering - Gurobi Direct Solve")
    print("=" * 60)

    # Load instance
    data = load_instance(args.instance_path)
    n_pairings = len(data["pairings"])
    n_crew = len(data["crew_members"])
    print(f"Instance: {n_crew} crew members, {n_pairings} pairings")
    print(f"Planning horizon: {data['planning_horizon_days']} days")
    print(f"Time limit: {args.time_limit} seconds")

    # Enumerate rosters for each crew member
    start_time = time.time()
    all_rosters = {}

    print(f"\nEnumerating feasible rosters...")
    data['_enumeration_budget']=max(.05,args.time_limit*.25/max(1,n_crew))
    for k, crew in enumerate(data["crew_members"]):
        rosters = build_network_and_enumerate(data, crew)
        all_rosters[k] = rosters
        print(f"  Crew {k}: {len(rosters)} feasible rosters "
              f"(pre-assigned days: {crew['pre_assigned_days']})")

    enum_time = time.time() - start_time
    total_rosters = sum(len(r) for r in all_rosters.values())
    print(f"Total rosters enumerated: {total_rosters} in {enum_time:.1f}s")

    # Adjust time limit for solver (subtract enumeration time)
    remaining_time = max(.1, args.time_limit - enum_time - 3)
    print(f"Remaining time for Gurobi: {remaining_time}s")

    # Build and solve model
    print("\nBuilding and solving Gurobi model...")
    solution = solve_model(data, all_rosters, remaining_time)

    # Output
    print(f"\nSolution status: {solution['status']}")
    if solution["objective_value"] is not None:
        print(f"Objective value (total uncovered pairing duration): {solution['objective_value']:.1f}")
        print(f"Covered pairings: {len(solution.get('covered_pairings', []))} / {n_pairings}")
        print(f"Uncovered pairings: {len(solution.get('uncovered_pairings', []))}")
        for up in solution.get("uncovered_pairings", []):
            print(f"  Pairing {up['pairing_id']}: {up['duration_minutes']} min uncovered")

    # Save solution
    with open(args.solution_path, 'w') as f:
        solution["runtime"] = time.time() - _GUROBI_CODE_START_TIME
        json.dump(solution, f, indent=2, default=str)
    print(f"\nSolution saved to: {args.solution_path}")

    total_time = time.time() - start_time
    print(f"Total time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
