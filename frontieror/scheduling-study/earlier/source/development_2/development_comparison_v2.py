"""Second development comparison on twelve fixed synthetic cases.

One execution per solver and case. This is development evidence, not the final
thirty-instance, three-execution confirmatory experiment.
"""

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET

from development_gate import evaluate_gate, GATE
from campaign import ROOT, belief, edit, read, register, save
from comparator_timing_calibration import IMAGE, validate_snapshots
from lisa_adapter import read_instance, write_instance
from synthetic_instances_v2 import VERSION, development_plan, generate


def main():
    current = ROOT / "campaign/comparator-setup/development-v2-current.json"
    # An exclusive marker prevents accidental duplicate launch, including a race.
    current.parent.mkdir(parents=True, exist_ok=True)
    with current.open("x") as marker:
        json.dump(dict(pid=os.getpid(), state="initializing"), marker)
    canary = json.loads((ROOT / "campaign/comparator-setup/timing-canary-current.json").read_text())
    if canary.get("state") != "complete" or canary.get("passed") is not True:
        raise RuntimeError("The single timing diagnostic must pass first")
    canary_record = read(canary["experiment_id"])
    assert canary_record["self"]["status"] == "complete"
    assert json.loads(canary_record["self"]["outcome"])["passed"] is True
    read(canary["belief_id"])
    folder = ROOT / "evidence" / f"development-comparison-v2-{time.time_ns()}"
    folder.mkdir()
    plan = development_plan()
    config = dict(stage="development", image=IMAGE, generator_version=VERSION, plan=plan,
                  external_deadline_seconds=300, internal_seconds={"egrasp": 290, "frozen-v2": 290},
                  executions_per_solver_case=1, seed_v2=0, seed_egrasp="undocumented; not seed-matched",
                  cpu_limit=1, memory="512m", capture_poll_seconds=.01,
                  rationale="Harder development distribution after eight zero ties; fixed 290s internal and 300s external budgets, all failures and zeros retained",
                  final_holdout_generated=False, python_version=platform.python_version(),
                  weights_priority="Exactly six uniformly sampled jobs of30 have weight10; others1",
                  due_dates="release + ceil(TF times own processing sum); exact rational TF3/2,7/4,2",
                  release_dates="uniform integers0..floor(total processing/(2n))",
                  provenance="LiSA handbook p70 equations7.1/7.2 motivate time formulas; other parameters are custom design")
    config.update(decision_gate=GATE, timing_canary=canary, host_concurrency="Sagent research continues separately; one CPU quota does not isolate the host", timing_claim="Equal external deadlines and nominal internal requests, not identical CPU effort")
    snapshots = folder / "source"
    snapshots.mkdir()
    hashes = {}
    for name in ["synthetic_instances_v2.py", "development_comparison_v2.py", "development_gate.py", "comparator_timing_calibration.py", "timed_solver.py", "lisa_adapter.py", "audit_schedule.py"]:
        shutil.copy2(ROOT / "pilot" / name, snapshots / name)
        hashes[name] = hashlib.sha256((snapshots / name).read_bytes()).hexdigest()
    config["source_sha256"] = hashes
    config["official_checker_sha256"] = hashlib.sha256((ROOT / "data/tasks/bierwirth2017/feasibility_check.py").read_bytes()).hexdigest()
    integrity = json.loads((ROOT / "evidence/development-integrity-1791444536066580000/verified.json").read_text())
    assert config["official_checker_sha256"] == integrity["official_checker_sha256"]
    assert config["image"] == integrity["image"]
    config["solver_sha256"] = integrity["binaries"]
    save(folder / "plan-before-generation.json", config)
    eid = register("Experiment", dict(title="Frozen v2 versus EGRASP: second twelve-case development batch, tighter due dates",
                   description="Codex executes the development design reviewed with ChatGPT. No final-test claim. All planned cases retained, including failure or zero objective.",
                   status="active", config=config), folder / "trackinizer")
    state = dict(pid=os.getpid(), evidence=str(folder), experiment_id=eid, state="generating", completed=0, planned=24)
    save(current, state)
    options = ET.parse(ROOT / "evidence/egrasp-compatibility-1791437962807856000/input.xml").getroot().find("controls")
    manifest = []
    for entry in plan:
        instance = generate(entry["seed"], entry["tightness"], entry["weights"])
        target = folder / "instances" / entry["name"]
        save(target / "instance.json", instance)
        xml = write_instance(instance, options)
        (target / "input.xml").write_text(xml)
        assert read_instance(ET.fromstring(xml)) == instance
        lines = [f"{instance['num_jobs']} {instance['num_machines']}"]
        for job in instance["jobs"]:
            lines.append(" ".join(str(job[k]) for k in ["job_id", "release_date", "due_date", "weight"]))
            lines.extend(f"{op['machine']} {op['processing_time']}" for op in job["operations"])
        (target / "input.txt").write_text("\n".join(lines) + "\n")
        tokens = iter(map(int, (target / "input.txt").read_text().split()))
        n, m = next(tokens), next(tokens)
        jobs = []
        for _ in range(n):
            job = dict(zip(["job_id", "release_date", "due_date", "weight"], [next(tokens) for _ in range(4)]))
            job["operations"] = [dict(machine=next(tokens), processing_time=next(tokens)) for _ in range(m)]
            jobs.append(job)
        assert not list(tokens) and n == 30 and m == 10 and jobs == instance["jobs"]
        manifest.append(dict(**entry, sha256={name: hashlib.sha256((target / name).read_bytes()).hexdigest() for name in ["instance.json", "input.xml", "input.txt"]}))
    save(folder / "instance-manifest-before-execution.json", manifest)
    container = f"frontieror-development-v2-{time.time_ns()}"
    command = ["docker", "run", "-d", "--rm", "--name", container, "--platform", "linux/amd64",
               "--network", "none", "--cpus", "1", "--memory", "512m", "--cap-drop", "ALL",
               "--security-opt", "no-new-privileges", "--mount", f"type=bind,source={folder},target=/work", IMAGE, "sleep", "infinity"]
    save(folder / "container-command.json", command)
    results = []
    try:
        launched = subprocess.run(command, capture_output=True, text=True, check=True, timeout=30)
        state.update(container=container, container_id=launched.stdout.strip())
        save(folder / "container-inspect.json", json.loads(subprocess.check_output(["docker", "inspect", container], text=True)))
        binaries = subprocess.check_output(["docker", "exec", container, "sha256sum", "egrasp", "frozen-v2", "frozen-v2.cpp", "timed_solver.py"], text=True)
        (folder / "container-source-sha256.txt").write_text(binaries)
        actual = {line.split()[1]: line.split()[0] for line in binaries.splitlines()}
        assert all(actual[name] == sha for name, sha in config["solver_sha256"].items())
        assert actual["timed_solver.py"] == config["source_sha256"]["timed_solver.py"]
        for index, entry in enumerate(plan):
            solvers = ["frozen-v2", "egrasp"] if index % 2 == 0 else ["egrasp", "frozen-v2"]
            for solver in solvers:
                case = folder / "runs" / f"{entry['name']}-{solver}"
                case.mkdir(parents=True)
                for name in ["instance.json", "input.xml", "input.txt"]:
                    shutil.copy2(folder / "instances" / entry["name"] / name, case / name)
                instance = json.loads((case / "instance.json").read_text())
                inside = "/work/runs/" + case.name
                output = "output.xml" if solver == "egrasp" else "output.json"
                argv = ["./egrasp", inside + "/input.xml", inside + "/" + output, "s", "290"] if solver == "egrasp" else ["./frozen-v2", inside + "/input.txt", inside + "/" + output, "290", "0", "1"]
                runner = ["docker", "exec", container, "python3", "timed_solver.py", "--output", inside + "/" + output,
                          "--directory", inside + "/timing", "--deadline", "300", "--", *argv]
                save(case / "command.json", dict(argv=runner, started_unix=time.time()))
                state.update(state="running", current=case.name, current_started_unix=time.time())
                save(current, state)
                process = subprocess.run(runner, capture_output=True, text=True, timeout=330)
                (case / "launcher-stdout.txt").write_text(process.stdout)
                (case / "launcher-stderr.txt").write_text(process.stderr)
                measurement = json.loads((case / "timing/measurement.json").read_text())
                checks = validate_snapshots(case, instance, solver, measurement)
                valid = [x for x in checks if x.get("valid") and x["observed_seconds"] <= 300]
                stdout = (case / "timing/stdout.txt").read_text()
                reported = re.findall(r"best solution with TWT value:\s*([\d.]+)", stdout) if solver == "egrasp" else []
                # Never treat contradictory binary/report semantics as a verified comparison.
                agrees = (not any(x.get("error") for x in checks) and
                          (solver != "egrasp" or not valid or
                           (bool(reported) and float(reported[-1]) == valid[-1]["objective"])))
                row = dict(instance=entry["name"], stratum=f"tf{entry['tightness']}-{entry['weights']}", solver=solver,
                           status=("VALID_IN_TIME" if valid else "SOLVER_FAILURE") if agrees else "UNRESOLVED_SEMANTICS",
                           objective=min(x["objective"] for x in valid) if valid and agrees else None,
                           measurement=measurement, checks=checks, report_matches_snapshot=agrees,
                           reported=float(reported[-1]) if reported else None)
                save(case / "result.json", row)
                results.append(row)
                outcome = dict(state="partial", config=config, manifest=manifest, results=results)
                save(folder / "partial-results.json", outcome)
                edit(eid, "outcome", json.dumps(outcome))
                verified = read(eid)
                assert json.loads(verified["self"]["outcome"]) == outcome
                state.update(completed=len(results), latest=dict(instance=entry["name"], solver=solver, status=row["status"], objective=row["objective"]))
                save(current, state)
                print(json.dumps(state["latest"]), flush=True)
        gate = evaluate_gate(plan, results, canary_passed=True)
        save(folder / "decision-gate.json", gate)
        outcome = dict(state="complete", config=config, manifest=manifest, results=results, gate=gate)
        save(folder / "results.json", outcome)
        edit(eid, "outcome", json.dumps(outcome))
        edit(eid, "status", "complete")
        verified = read(eid)
        assert json.loads(verified["self"]["outcome"]) == outcome
        save(folder / "trackinizer/outcome-readback.json", verified)
        compact = [{k: row[k] for k in ["instance", "solver", "status", "objective"]} for row in results]
        bid = belief("Second synthetic development comparison observations", "One execution per solver and case; development only. " + json.dumps(dict(gate=gate, results=compact)), [eid], folder / "belief")
        state.update(state="complete", gate=gate, belief_id=bid, finished_unix=time.time())
        save(current, state)
        with (ROOT / "JOURNEY.md").open("a") as journal:
            journal.write(f"\n## Second synthetic development comparison completed\n\nThe12 cases and settings were recorded before execution. All24 planned executions are preserved at `{folder.relative_to(ROOT)}`, including failures and zeros. Experiment `{eid}` and observation `{bid}` stored and read back. One execution per solver/case is development evidence, not the planned final test. Fixed difficulty and reliability gate: `{gate}`. Final test generation requires separate protocol freeze after audit.\n")
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
