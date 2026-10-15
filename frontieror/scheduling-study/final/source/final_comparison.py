"""Frozen final comparison: thirty fresh cases, three repetitions per solver.

No automatic restart or retry. All attempts are retained. Intermediate metrics
are saved locally; Trackinizer receives commitments until the full batch ends.
"""

import hashlib
import json
import os
import platform
from pathlib import Path
import re
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET

from final_protocol import instance_plan, execution_plan, analysis_records, SOLVER_SEEDS, ORDER_SEED, BOOTSTRAP_SEED
from comparator_analysis import analyze
from campaign import ROOT, belief, edit, read, register, save
from comparator_timing_calibration import IMAGE, validate_snapshots
from lisa_adapter import read_instance, write_instance
from synthetic_instances_v2 import VERSION, generate


def main():
    current = ROOT / "campaign/comparator-setup/final-current.json"
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
    folder = ROOT / "evidence" / f"final-comparison-{time.time_ns()}"
    folder.mkdir()
    development = json.loads((ROOT / "campaign/comparator-setup/development-v2-current.json").read_text())
    audit_path = ROOT / "evidence/development-v2-final-audit-1791453271636198000/verified-summary.json"
    audit = json.loads(audit_path.read_text())
    assert development["state"] == "complete" and development["gate"]["passed"]
    assert audit["state"] == "verified" and audit["executions"] == 24 and audit["valid"] == 24
    assert audit["experiment_id"] == development["experiment_id"]
    assert read(development["experiment_id"])["self"]["status"] == "complete"
    read(audit["audit_belief"])
    plan = instance_plan()
    schedule = execution_plan()
    config = dict(stage="confirmatory", image=IMAGE, generator_version=VERSION, plan=plan,
                  external_deadline_seconds=300, internal_seconds={"egrasp": 290, "frozen-v2": 290},
                  executions_per_solver_case=3, seeds_v2=list(SOLVER_SEEDS), seed_egrasp="undocumented; not seed-matched",
                  cpu_limit=1, memory="512m", capture_poll_seconds=.01,
                  rationale="Final evaluation on a fixed custom distribution selected during development; all zeros and failures retained, no input or implementation changes",
                  final_data_generated_at_registration=False, python_version=platform.python_version(),
                  weights_priority="Exactly six uniformly sampled jobs of30 have weight10; others1",
                  due_dates="release + ceil(TF times own processing sum); exact rational TF3/2,7/4,2",
                  release_dates="uniform integers0..floor(total processing/(2n))",
                  provenance="LiSA handbook p70 equations7.1/7.2 motivate time formulas; other parameters are custom design")
    config.update(execution_schedule=schedule, order_seed=ORDER_SEED,
                  bootstrap_seed=BOOTSTRAP_SEED, bootstrap_resamples=10000,
                  development_experiment=development["experiment_id"], development_audit=audit,
                  timing_canary=canary,
                  primary="Mean of 30 instance-level H values, each averaging nine cross-repetition comparisons. Valid beats failure; both failed tie; lower objective wins.",
                  decision="Primary advantage only if two-sided95% stratified instance bootstrap lower bound for H is greater than zero AND candidate valid count >= comparator valid count. Otherwise not established.",
                  secondary="Mean (mean E - mean V)/max(1,mean E,mean V). Full-sample numeric analysis only if all180 executions valid; otherwise conditional descriptive comparisons only.",
                  failure_policy="Solver failure has no retry. Host/runner interruption or unresolved semantics stops this evaluation and retains every attempt. No automatic replacements; any continuation requires documented review before a completion claim.",
                  masking="Do not feed intermediate scores to Sagent or change the frozen solvers, input plan, schedule or analysis. Trackinizer stores progress and file commitments until completion. Shared host filesystem is not an access-control boundary.",
                  host_concurrency="Sagent continues separately; one CPU quota does not isolate the host",
                  timing_claim="Equal external deadlines and nominal internal requests, not equal CPU work",
                  claim_scope="This custom distribution and these frozen implementations under emulation; not algorithmic novelty, optimality, current SOTA or a FrontierOR-wide result")
    snapshots = folder / "source"
    snapshots.mkdir()
    hashes = {}
    for name in ["synthetic_instances_v2.py", "final_comparison.py", "final_protocol.py", "comparator_analysis.py", "comparator_timing_calibration.py", "timed_solver.py", "lisa_adapter.py", "audit_schedule.py"]:
        shutil.copy2(ROOT / "pilot" / name, snapshots / name)
        hashes[name] = hashlib.sha256((snapshots / name).read_bytes()).hexdigest()
    config["source_sha256"] = hashes
    config["official_checker_sha256"] = hashlib.sha256((ROOT / "data/tasks/bierwirth2017/feasibility_check.py").read_bytes()).hexdigest()
    integrity = json.loads((ROOT / "evidence/development-integrity-1791444536066580000/verified.json").read_text())
    assert config["official_checker_sha256"] == integrity["official_checker_sha256"]
    assert config["image"] == integrity["image"]
    config["solver_sha256"] = integrity["binaries"]
    save(folder / "plan-before-generation.json", config)
    eid = register("Experiment", dict(title="Frozen v2 versus historical EGRASP: preregistered thirty-case final evaluation",
                   description="Codex runs the fixed final design after the independently audited development gates passed. All180 planned executions retained. ChatGPT advised methodology; user authorized the research. No intermediate tuning or automatic replacement.",
                   status="active", config=config), folder / "trackinizer")
    assert read(eid)["self"]["config"] == config
    state = dict(pid=os.getpid(), evidence=str(folder), experiment_id=eid, state="generating", completed=0, planned=180)
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
    canonical_hashes = [entry["sha256"]["instance.json"] for entry in manifest]
    prior_manifest = json.loads((Path(development["evidence"]) / "instance-manifest-before-execution.json").read_text())
    prior_hashes = {entry["sha256"]["instance.json"] for entry in prior_manifest}
    assert len(set(canonical_hashes)) == 30 and not set(canonical_hashes) & prior_hashes
    save(folder / "instance-manifest-before-execution.json", manifest)
    container = f"frontieror-final-{time.time_ns()}"
    command = ["docker", "run", "-d", "--rm", "--name", container, "--platform", "linux/amd64",
               "--network", "none", "--cpus", "1", "--memory", "512m", "--cap-drop", "ALL",
               "--security-opt", "no-new-privileges", "--mount", f"type=bind,source={folder},target=/work", IMAGE, "sleep", "infinity"]
    save(folder / "container-command.json", command)
    results = []
    commitments = []
    try:
        launched = subprocess.run(command, capture_output=True, text=True, check=True, timeout=30)
        state.update(container=container, container_id=launched.stdout.strip())
        save(folder / "container-inspect.json", json.loads(subprocess.check_output(["docker", "inspect", container], text=True)))
        binaries = subprocess.check_output(["docker", "exec", container, "sha256sum", "egrasp", "frozen-v2", "frozen-v2.cpp", "timed_solver.py"], text=True)
        (folder / "container-source-sha256.txt").write_text(binaries)
        actual = {line.split()[1]: line.split()[0] for line in binaries.splitlines()}
        assert all(actual[name] == sha for name, sha in config["solver_sha256"].items())
        assert actual["timed_solver.py"] == config["source_sha256"]["timed_solver.py"]
        entries = {entry["name"]: entry for entry in plan}
        for slot in schedule:
            entry = entries[slot["instance"]]
            solver = slot["solver"]
            case = folder / "runs" / slot["slot"]
            case.mkdir(parents=True)
            for name in ["instance.json", "input.xml", "input.txt"]:
                shutil.copy2(folder / "instances" / entry["name"] / name, case / name)
            instance = json.loads((case / "instance.json").read_text())
            inside = "/work/runs/" + case.name
            output = "output.xml" if solver == "egrasp" else "output.json"
            argv = ["./egrasp", inside + "/input.xml", inside + "/" + output, "s", "290"] if solver == "egrasp" else ["./frozen-v2", inside + "/input.txt", inside + "/" + output, "290", str(slot["seed"]), "1"]
            runner = ["docker", "exec", container, "python3", "timed_solver.py", "--output", inside + "/" + output,
                      "--directory", inside + "/timing", "--deadline", "300", "--", *argv]
            save(case / "command.json", dict(argv=runner, started_unix=time.time()))
            state.update(state="running", current=case.name, current_started_unix=time.time())
            save(current, state)
            started = dict(state="running", completed=len(results), planned=180,
                           active_slot=slot, started_unix=state["current_started_unix"], commitments=commitments)
            edit(eid, "outcome", json.dumps(started))
            assert json.loads(read(eid)["self"]["outcome"]) == started
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
            row = dict(**slot, stratum=f"tf{entry['tightness']}-{entry['weights']}",
                       status=("VALID_IN_TIME" if valid else "SOLVER_FAILURE") if agrees else "UNRESOLVED_SEMANTICS",
                       objective=min(x["objective"] for x in valid) if valid and agrees else None,
                       measurement=measurement, checks=checks, report_matches_snapshot=agrees,
                       reported=float(reported[-1]) if reported else None)
            save(case / "result.json", row)
            results.append(row)
            outcome = dict(state="partial", config=config, manifest=manifest, results=results)
            save(folder / "partial-results.json", outcome)
            commitment = dict(state="running", completed=len(results), planned=180,
                              slot=slot["slot"], result_sha256=hashlib.sha256((case / "result.json").read_bytes()).hexdigest(),
                              cumulative_sha256=hashlib.sha256((folder / "partial-results.json").read_bytes()).hexdigest(),
                              performance="Withheld from research agent until fixed batch completion")
            save(case / "trackinizer-commitment.json", commitment)
            commitments.append(commitment)
            progress = dict(state="running", completed=len(results), planned=180, commitments=commitments)
            edit(eid, "outcome", json.dumps(progress))
            verified = read(eid)
            assert json.loads(verified["self"]["outcome"]) == progress
            save(case / "trackinizer-readback.json", verified)
            state.update(completed=len(results), latest_slot=slot["slot"])
            save(current, state)
            print(json.dumps(commitment), flush=True)
            if row["status"] == "UNRESOLVED_SEMANTICS":
                raise RuntimeError("Unresolved output semantics; final evaluation stopped for review")
        records = analysis_records(plan, results)
        analysis = analyze(records, BOOTSTRAP_SEED)
        save(folder / "analysis.json", analysis)
        outcome = dict(state="complete", config=config, manifest=manifest, results=results, analysis=analysis)
        save(folder / "results.json", outcome)
        edit(eid, "outcome", json.dumps(outcome))
        edit(eid, "status", "complete")
        verified = read(eid)
        assert json.loads(verified["self"]["outcome"]) == outcome
        save(folder / "trackinizer/outcome-readback.json", verified)
        bid = belief("Final synthetic comparison: measured outcome", "All180 planned executions retained. This is a comparison of frozen implementations on a preregistered custom distribution, not current SOTA. " + json.dumps(analysis), [eid], folder / "belief")
        state.update(state="complete", belief_id=bid, finished_unix=time.time())
        save(current, state)
        with (ROOT / "JOURNEY.md").open("a") as journal:
            journal.write(f"\n## Final synthetic comparison completed\n\nAll180 executions preserved at `{folder.relative_to(ROOT)}`. Experiment `{eid}` and observation `{bid}` stored and read back. Statistical analysis saved; independent end-to-end audit still required before a final claim.\n")
        print(json.dumps(state), flush=True)
    except Exception as exc:
        state.update(state="incomplete_requires_review", error=repr(exc))
        save(current, state)
        save(folder / "failure.json", dict(state=state, results=results))
        try:
            edit(eid, "outcome", json.dumps(dict(state="incomplete_requires_review", completed=len(results), error=repr(exc), evidence=str(folder), active_slot=state.get("current"), commitments=commitments, no_automatic_retry=True)))
        except Exception as tracking_error:
            save(folder / "tracking-failure.json", dict(error=repr(tracking_error)))
        raise
    finally:
        subprocess.run(["docker", "stop", "--time", "1", container], capture_output=True, timeout=15)


if __name__ == "__main__":
    main()
