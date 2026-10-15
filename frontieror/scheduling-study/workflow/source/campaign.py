"""Local, recorded research campaign. Requires a reachable Trackinizer server."""

import ast
import hashlib
import json
import os
import signal
import subprocess
import threading
import time
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/tasks/bierwirth2017"
PYTHON = ROOT / ".venv/bin/python"
URL = "[private-endpoint-removed]"
ACTOR = "Venkat/Codex campaign runner"
LOCK = threading.RLock()
CONTRACT = {
    "solver_seconds": 10,
    "hint_seconds_max": 2,
    "process_seconds_max": 30,
    "search_workers": 1,
    "development_instances": ["tiny", "large-1"],
    "objective": "minimize total weighted tardiness",
    "acceptance": "official checker AND independent audit",
    "memory": "host memory; no competition memory sandbox",
    "cpu": "one CP-SAT worker and single-thread hint, no CPU affinity",
    "scope": "Change hint construction only. CP-SAT model/settings fixed.",
}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def request(path, body=None, method=None):
    req = urllib.request.Request(
        URL + path,
        data=None if body is None else json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method=method or ("GET" if body is None else "POST"),
    )
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)


def root_id():
    return json.loads((ROOT / "campaign/root.json").read_text())["id"]


def read(record_id):
    return request("/api/web/get/" + record_id)


def register(kind, body, receipt):
    receipt.mkdir(parents=True, exist_ok=True)
    body = {"actor": ACTOR, "idempotency_key": str(uuid.uuid4()), **body}
    save(receipt / "request.json", body)
    record_id = request("/api/inquiries/" + kind.lower(), body)["id"]
    save(receipt / "created.json", {"id": record_id, "kind": kind})
    request(f"/api/edges/{record_id}/produced_by/{root_id()}", {"actor": ACTOR})
    verified = read(record_id)
    assert verified["self"]["title"] == body["title"]
    if "outcome" in body:
        assert verified["self"]["outcome"] == body["outcome"]
    save(receipt / "readback.json", verified)
    return record_id


def edit(record_id, field, value):
    prefix = "experiment" if field in {"outcome", "config"} else "inquiries"
    request(
        f"/api/{prefix}/{record_id}/{field}", {"value": value, "actor": ACTOR}, "PUT"
    )


def belief(title, description, experiment_ids, folder):
    record_id = register(
        "Belief",
        {
            "title": title,
            "description": description,
            "judgement": "unproven",
            "status": "active",
            "favored_by": [
                {"artifact_id": eid, "artifact_kind": "Experiment", "valence": 1.0}
                for eid in experiment_ids
            ],
        },
        folder,
    )
    detail = read(record_id)
    # Verify the actual stored evidence references, not only the text.
    encoded = json.dumps(detail["edges"]) + json.dumps(detail["backlinks"])
    assert all(eid in encoded for eid in experiment_ids), detail
    save(folder / "evidence-readback.json", detail)
    return record_id


def import_history():
    read(root_id())  # Must succeed before any work.
    target = ROOT / "campaign/imports"
    for index, old in enumerate(
        json.loads((ROOT / "baseline-checks.json").read_text())
    ):
        folder = target / f"setup-{index:02d}"
        if (folder / "complete.json").exists():
            continue
        eid = register(
            "Experiment",
            {
                "title": f"Setup: {old['solver']} on {old['instance']}",
                "description": "Imported prior setup measurement, orchestrated by Codex before this campaign. Not a Sagent proposal. Original evidence is preserved.",
                "status": "complete",
                "outcome": json.dumps(old),
                "config": old,
            },
            folder / "experiment",
        )
        bid = belief(
            f"Setup observation {index}: validity={old['feasible']}",
            "Observed result only; does not establish general performance or a new best known result. "
            + json.dumps(old),
            [eid],
            folder / "belief",
        )
        save(folder / "complete.json", {"experiment": eid, "belief": bid})
    save(ROOT / "campaign/evaluation-contract.json", CONTRACT)


def run(argv, cwd, folder, timeout):
    folder.mkdir(parents=True, exist_ok=False)
    env = {
        key: os.environ[key] for key in ("PATH", "TMPDIR", "LANG") if key in os.environ
    }
    env.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "PYTHONHASHSEED": "0",
        }
    )
    started = time.time()
    info = {
        "argv": list(map(str, argv)),
        "cwd": str(cwd),
        "environment": env,
        "started_unix": started,
        "deadline_seconds": timeout,
    }
    save(folder / "command.json", info)
    with (
        (folder / "stdout.txt").open("w") as out,
        (folder / "stderr.txt").open("w") as err,
    ):
        process = subprocess.Popen(
            list(map(str, argv)),
            cwd=cwd,
            env=env,
            stdout=out,
            stderr=err,
            start_new_session=True,
        )
        timed_out = False
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
    info.update(
        {
            "elapsed_seconds": time.time() - started,
            "returncode": process.returncode,
            "timed_out": timed_out,
        }
    )
    save(folder / "command.json", info)
    return info


def check_source(source):
    """Restrict accidental I/O in hint proposals; not a security sandbox."""
    tree = ast.parse(source)
    allowed = {
        "math",
        "random",
        "time",
        "statistics",
        "heapq",
        "itertools",
        "collections",
        "functools",
        "copy",
    }
    blocked = {
        "open",
        "exec",
        "eval",
        "compile",
        "__import__",
        "globals",
        "locals",
        "getattr",
        "setattr",
        "vars",
        "input",
        "breakpoint",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            assert all(item.name.split(".")[0] in allowed for item in node.names), (
                "Only scheduling stdlib imports are allowed"
            )
        if isinstance(node, ast.ImportFrom):
            assert not node.level and node.module.split(".")[0] in allowed, (
                "Unsupported import"
            )
        if isinstance(node, ast.Name):
            assert node.id not in blocked, "Hint must compute from the input only"
        if isinstance(node, ast.Attribute):
            assert not node.attr.startswith("__"), "Dunder introspection is not allowed"


def evaluate(
    source, hypothesis, label, seed=0, actor="Sagent", instances=("tiny", "large-1")
):
    with LOCK:
        read(root_id())  # A failed read prevents source execution.
        folder = ROOT / "campaign/experiments" / label
        folder.mkdir(parents=True, exist_ok=False)
        digest = hashlib.sha256(source.encode()).hexdigest()
        (folder / "hint_proposal.py").write_text(source)
        config = {
            **CONTRACT,
            "seed": seed,
            "idea_source": actor,
            "source_sha256": digest,
            "source_path": str(folder / "hint_proposal.py"),
            "instances": list(instances),
        }
        eid = register(
            "Experiment",
            {"title": label, "description": hypothesis, "config": config},
            folder / "trackinizer",
        )
        results = []
        try:
            if actor == "Sagent":
                check_source(source)
            for name in instances:
                case = folder / name
                case.mkdir()
                filename = (
                    "tiny_instance.json"
                    if name == "tiny"
                    else f"large_instance_{name.split('-')[1]}.json"
                )
                original = DATA / "instance" / filename
                (case / "instance.json").write_bytes(original.read_bytes())
                (case / "hint_proposal.py").write_text(source)
                # Only this wrapper replaces the baseline's import. CP-SAT remains unchanged.
                (case / "dispatch_baseline.py").write_text(
                    "import json, time\nfrom hint_proposal import solve as proposal\n"
                    "def solve(instance):\n    start=time.monotonic()\n    result=proposal(instance)\n"
                    "    elapsed=time.monotonic()-start\n"
                    "    with open('hint.json','w') as f: json.dump(result,f)\n"
                    "    with open('hint-time.json','w') as f: json.dump({'seconds':elapsed},f)\n"
                    "    if elapsed>2: raise RuntimeError('Hint exceeded 2-second budget')\n"
                    "    return result\n"
                )
                (case / "cpsat_baseline.py").write_bytes(
                    (ROOT / "pilot/cpsat_baseline.py").read_bytes()
                )
                info = run(
                    [
                        PYTHON,
                        case / "cpsat_baseline.py",
                        "--instance_path",
                        case / "instance.json",
                        "--solution_path",
                        case / "solution.json",
                        "--metadata_path",
                        case / "metadata.json",
                        "--time_limit",
                        "10",
                        "--seed",
                        str(seed),
                    ],
                    case,
                    case / "execution",
                    30,
                )
                result = {
                    "instance": name,
                    "seed": seed,
                    "elapsed_seconds": info["elapsed_seconds"],
                    "returncode": info["returncode"],
                    "timed_out": info["timed_out"],
                    "feasible": False,
                }
                if info["returncode"] == 0:
                    for tag, script in (
                        ("official", DATA / "feasibility_check.py"),
                        ("audit", ROOT / "pilot/audit_schedule.py"),
                    ):
                        check = run(
                            [
                                PYTHON,
                                script,
                                "--instance_path",
                                case / "instance.json",
                                "--solution_path",
                                case / "solution.json",
                                "--result_path",
                                case / f"{tag}.json",
                            ],
                            case,
                            case / tag,
                            15,
                        )
                        if check["returncode"] != 0:
                            raise RuntimeError(f"{tag} check crashed")
                        result[tag] = json.loads((case / f"{tag}.json").read_text())
                    result["metadata"] = json.loads(
                        (case / "metadata.json").read_text()
                    )
                    result["metadata"]["hint"] = (
                        digest  # Upstream baseline description is fixed; identify actual hint here.
                    )
                    result["hint_seconds"] = json.loads(
                        (case / "hint-time.json").read_text()
                    )["seconds"]
                    result["hint_objective"] = json.loads(
                        (case / "hint.json").read_text()
                    )["objective_value"]
                    result["feasible"] = (
                        result["official"]["feasible"] and result["audit"]["feasible"]
                    )
                    result["objective"] = (
                        result["audit"]["recomputed_objective"]
                        if result["feasible"]
                        else None
                    )
                else:
                    result["error"] = (case / "execution/stderr.txt").read_text()[
                        -4000:
                    ]
                results.append(result)
        except Exception as exc:  # noqa: BLE001 - persist failed experiments before returning
            results.append({"error": f"{type(exc).__name__}: {exc}", "feasible": False})
        summary = {
            "label": label,
            "experiment_id": eid,
            "hypothesis": hypothesis,
            "source_sha256": digest,
            "results": results,
            "config": config,
        }
        save(folder / "results.json", summary)
        outcome = json.dumps(summary)
        edit(eid, "outcome", outcome)
        edit(eid, "status", "complete")
        detail = read(eid)
        assert (
            detail["self"]["outcome"] == outcome
            and detail["self"]["status"] == "complete"
        )
        save(folder / "trackinizer/completed-readback.json", detail)
        brief = [
            {
                k: r[k]
                for k in (
                    "instance",
                    "seed",
                    "feasible",
                    "objective",
                    "hint_objective",
                    "elapsed_seconds",
                    "error",
                )
                if k in r
            }
            for r in results
        ]
        bid = belief(
            f"Measured outcome: {label}",
            "These are measurements under the recorded limits, not evidence of generalization. "
            + json.dumps(brief),
            [eid],
            folder / "observation",
        )
        summary["observation_id"] = bid
        save(folder / "verified.json", summary)
        with (ROOT / "JOURNEY.md").open("a") as journal:
            journal.write(
                f"\n## 7 October 2026: {label}\n\nHypothesis: {hypothesis}\n\nIdea source: {actor}. Code SHA-256: `{digest}`. CP-SAT: 10 seconds, one worker, seed {seed}; hint at most 2 seconds; whole solver process at most 30 seconds.\n\nMeasured results: `{json.dumps(brief)}`.\n\nEvidence: `campaign/experiments/{label}/`. Trackinizer experiment `{eid}` and observation `{bid}` were read back before the next evaluation. Interpretation requires paired repeats before a broader claim.\n"
            )
        print(
            json.dumps({"recorded_experiment": label, "id": eid, "results": brief}),
            flush=True,
        )
        return summary


def state():
    root = read(root_id())
    records = []
    # IDs are an index; the contents passed to Sagent are fetched from Trackinizer.
    for path in sorted((ROOT / "campaign").glob("**/created.json")):
        meta = json.loads(path.read_text())
        row = read(meta["id"])["self"]
        records.append(
            {
                key: row.get(key)
                for key in ("id", "kind", "title", "description", "status", "outcome")
            }
        )
    return {"campaign": root["self"], "contract": CONTRACT, "records": records}


def export(path):
    with urllib.request.urlopen(URL + "/api/export", timeout=30) as response:
        path.write_bytes(response.read())
