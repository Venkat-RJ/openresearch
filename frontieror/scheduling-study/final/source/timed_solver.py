"""Measure one solver process, preserving only parseable artifacts observed in time.

Run inside the common container. XML/JSON parsing here establishes completeness,
not schedule validity. Both schedule validators must still run afterward.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import time
import xml.etree.ElementTree as ET


def measure(argv, output, directory, deadline, poll_seconds=0.01):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    output = Path(output)
    if output.exists():
        raise ValueError("Refuse preexisting solver output")
    if deadline <= 0:
        raise ValueError("Positive deadline required")
    samples, hashes = [], set()
    before_cpu = resource.getrusage(resource.RUSAGE_CHILDREN)
    started_unix = time.time()
    start = time.monotonic()
    end = start + deadline
    timed_out = False
    with (directory / "stdout.txt").open("w") as stdout, (directory / "stderr.txt").open("w") as stderr:
        child = subprocess.Popen(argv, stdout=stdout, stderr=stderr, start_new_session=True)

        def capture():
            if time.monotonic() >= end:
                return
            try:
                content = output.read_bytes()
            except FileNotFoundError:
                return
            observed = time.monotonic() - start
            if observed > deadline:
                return
            digest = hashlib.sha256(content).hexdigest()
            if digest in hashes:
                return
            hashes.add(digest)
            error = None
            try:
                if output.suffix == ".xml":
                    ET.fromstring(content)
                else:
                    json.loads(content)
            except (ValueError, ET.ParseError, UnicodeError) as exc:
                error = type(exc).__name__
            row = dict(observed_seconds=observed, sha256=digest, bytes=len(content),
                       parseable=error is None, parse_error=error)
            if error is None:
                filename = f"snapshot-{len(samples):04d}{output.suffix}"
                (directory / filename).write_bytes(content)
                row["snapshot"] = filename
            samples.append(row)

        while True:
            capture()
            code = child.poll()
            if code is not None:
                capture()
                break
            remaining = end - time.monotonic()
            if remaining <= 0:
                timed_out = True
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                code = child.wait(timeout=5)
                break
            time.sleep(min(poll_seconds, remaining))
    elapsed = time.monotonic() - start
    after_cpu = resource.getrusage(resource.RUSAGE_CHILDREN)
    result = dict(argv=argv, pid=child.pid, started_unix=started_unix,
                  external_deadline_seconds=deadline, elapsed_seconds=elapsed,
                  user_cpu_seconds=after_cpu.ru_utime - before_cpu.ru_utime,
                  system_cpu_seconds=after_cpu.ru_stime - before_cpu.ru_stime,
                  returncode=code, timed_out=timed_out, samples=samples,
                  validation_status="Pending both schedule validators; parseable is not feasible",
                  timing_scope="Process startup and execution; excludes container startup and preconversion")
    (directory / "measurement.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--directory", required=True)
    parser.add_argument("--deadline", type=float, required=True)
    parser.add_argument("argv", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.argv[1:] if args.argv[:1] == ["--"] else args.argv
    print(json.dumps(measure(command, args.output, args.directory, args.deadline)))
