"""Calibrate process timing on public data. Never interpret as final performance."""

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET

from audit_schedule import audit
from campaign import ROOT, belief, edit, read, register, save
from lisa_adapter import read_solution

IMAGE = "sha256:5bf378efea878ad3fba6a3772e7eee8061dcd620d6f738e8dccd27286679c448"


def validate_snapshots(case, instance, solver, measurement):
    checks = []
    for sample in measurement["samples"]:
        if not sample["parseable"]:
            continue
        snapshot = case / "timing" / sample["snapshot"]
        row = dict(snapshot=sample["snapshot"], observed_seconds=sample["observed_seconds"], sha256=sample["sha256"])
        try:
            assert hashlib.sha256(snapshot.read_bytes()).hexdigest() == sample["sha256"]
            solution = read_solution(ET.parse(snapshot).getroot(), instance) if solver == "egrasp" else json.loads(snapshot.read_text())
            row["audit"] = audit(instance, solution)
            target = case / (snapshot.stem + "-validation")
            target.mkdir()
            save(target / "solution.json", solution)
            save(target / "audit.json", row["audit"])
            command = [str(ROOT / ".venv/bin/python"), str(ROOT / "data/tasks/bierwirth2017/feasibility_check.py"),
                       "--instance_path", str(case / "instance.json"), "--solution_path", str(target / "solution.json"),
                       "--result_path", str(target / "official.json")]
            save(target / "command.json", command)
            process = subprocess.run(command, capture_output=True, text=True, timeout=30)
            (target / "stdout.txt").write_text(process.stdout)
            (target / "stderr.txt").write_text(process.stderr)
            row["official"] = json.loads((target / "official.json").read_text())
            row["objective"] = row["audit"]["recomputed_objective"]
            row["valid"] = bool(row["audit"]["feasible"] and row["official"]["feasible"])
        except Exception as exc:
            row.update(valid=False, error=repr(exc))
        checks.append(row)
    return checks


def main():
    folder = ROOT / "evidence" / f"comparator-timing-calibration-{time.time_ns()}"
    folder.mkdir()
    source = ROOT / "evidence/comparator-scale-probe-1791438936820864000"
    plan = [["egrasp", seconds, repeat] for seconds, count in [(2, 5), (10, 5), (30, 3), (60, 3)] for repeat in range(count)]
    plan += [["frozen-v2", seconds, 0] for seconds in [2, 10, 30, 60]]
    config = dict(image=IMAGE, instance="public large-1", source=str(source), plan=plan,
                  cpu_limit=1, memory="512m", capture_poll_seconds=0.01,
                  external_deadline="requested internal seconds plus15s, calibration only",
                  attribution="Codex process calibration with ChatGPT review; not equal-time solver-quality evidence",
                  host_concurrency="Sagent uses another host CPU; no final benchmark claim",
                  snapshot_rule="Only complete parseable bytes observed before deadline, then both validators")
    eid = register("Experiment", dict(title="EGRASP and frozen v2 process timing calibration",
                   description=config["attribution"], status="active", config=config), folder / "trackinizer")
    state = dict(pid=os.getpid(), evidence=str(folder), experiment_id=eid, state="starting", completed=0, planned=len(plan))
    current = ROOT / "campaign/comparator-setup/timing-current.json"
    save(current, state)
    container = f"frontieror-calibration-{time.time_ns()}"
    argv = ["docker", "run", "-d", "--rm", "--name", container, "--platform", "linux/amd64",
            "--network", "none", "--cpus", "1", "--memory", "512m", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges", "--mount", f"type=bind,source={folder},target=/work",
            IMAGE, "sleep", "infinity"]
    save(folder / "container-command.json", argv)
    results = []
    try:
        started = subprocess.run(argv, capture_output=True, text=True, check=True, timeout=30)
        state["container"] = container
        state["container_id"] = started.stdout.strip()
        save(folder / "container-inspect.json", json.loads(subprocess.check_output(["docker", "inspect", container], text=True)))
        for index, (solver, seconds, repeat) in enumerate(plan):
            case = folder / f"{index:02d}-{solver}-{seconds}s-r{repeat}"
            case.mkdir()
            for name in (["instance.json", "input.xml"] if solver == "egrasp" else ["instance.json", "input.txt"]):
                shutil.copy2(source / solver / name, case / name)
            instance = json.loads((case / "instance.json").read_text())
            inside = "/work/" + case.name
            filename = "output.xml" if solver == "egrasp" else "output.json"
            command = ["./egrasp", inside + "/input.xml", inside + "/" + filename, "s", str(seconds)] if solver == "egrasp" else ["./frozen-v2", inside + "/input.txt", inside + "/" + filename, str(seconds), "0", "1"]
            runner = ["docker", "exec", container, "python3", "timed_solver.py", "--output", inside + "/" + filename,
                      "--directory", inside + "/timing", "--deadline", str(seconds + 15), "--", *command]
            save(case / "command.json", dict(argv=runner, started_unix=time.time()))
            state.update(state="running", current=case.name, started_unix=time.time())
            save(current, state)
            process = subprocess.run(runner, capture_output=True, text=True, timeout=seconds + 40)
            (case / "launcher-stdout.txt").write_text(process.stdout)
            (case / "launcher-stderr.txt").write_text(process.stderr)
            measurement = json.loads((case / "timing/measurement.json").read_text())
            checks = validate_snapshots(case, instance, solver, measurement)
            valid = [x for x in checks if x.get("valid")]
            reported_text = (case / "timing/stdout.txt").read_text()
            reported = re.findall(r"best solution with TWT value:\s*([\d.]+)", reported_text) if solver == "egrasp" else []
            row = dict(case=case.name, solver=solver, requested_seconds=seconds, repeat=repeat,
                       measurement=measurement, snapshots=checks, valid_output=bool(valid),
                       best_valid_objective=min(x["objective"] for x in valid) if valid else None,
                       reported=float(reported[-1]) if reported else None)
            if solver == "egrasp":
                row["report_matches_snapshot"] = bool(valid and row["reported"] == valid[-1]["objective"])
            save(case / "result.json", row)
            results.append(row)
            state.update(completed=len(results), latest=dict(case=case.name, wall=measurement["elapsed_seconds"], valid=bool(valid), returncode=measurement["returncode"]))
            save(current, state)
            edit(eid, "outcome", json.dumps(dict(state="partial", config=config, results=results)))
            print(json.dumps(state["latest"]), flush=True)
        outcome = dict(state="complete", config=config, results=results)
        save(folder / "results.json", outcome)
        edit(eid, "outcome", json.dumps(outcome))
        edit(eid, "status", "complete")
        verified = read(eid)
        assert json.loads(verified["self"]["outcome"]) == outcome
        save(folder / "trackinizer/outcome-readback.json", verified)
        compact = [{"case": x["case"], "wall": x["measurement"]["elapsed_seconds"],
                    "cpu": x["measurement"]["user_cpu_seconds"] + x["measurement"]["system_cpu_seconds"],
                    "valid": x["valid_output"], "returncode": x["measurement"]["returncode"],
                    "first_complete": next((s["observed_seconds"] for s in x["measurement"]["samples"] if s["parseable"]), None)} for x in results]
        bid = belief("Comparator timing calibration observations", json.dumps(dict(scope=config["attribution"], measurements=compact)), [eid], folder / "belief")
        state.update(state="complete", belief_id=bid, finished_unix=time.time())
        save(current, state)
        with (ROOT / "JOURNEY.md").open("a") as journal:
            journal.write(f"\n## Comparator process calibration\n\nCodex ran the preregistered development calibration plan in one persistent Linux x86 container. This measures process wall/CPU time and output availability, not equal-budget quality. Original commands, captured bytes, both validators and measurements: `{folder.relative_to(ROOT)}`. Experiment `{eid}` and observation `{bid}` were stored and read back. Next: derive a fixed deadline policy from these development measurements before final evaluation.\n")
        print(json.dumps(state), flush=True)
    except Exception as exc:
        state.update(state="failed", error=repr(exc))
        save(current, state)
        save(folder / "failure.json", dict(state=state, results=results))
        raise
    finally:
        subprocess.run(["docker", "stop", "--time", "1", container], capture_output=True, timeout=15)


if __name__ == "__main__":
    main()
