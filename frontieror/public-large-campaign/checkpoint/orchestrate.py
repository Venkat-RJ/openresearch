#!/usr/bin/env python3
"""Resumable canonical FrontierOR inventory, bounded download, official evaluation."""
import argparse
import ast
import concurrent.futures as futures
import csv
import datetime as dt
import difflib
from decimal import Decimal
import fcntl
import gzip
import importlib.util
import math
import hashlib
import json
import os
from pathlib import Path
import re
import psutil
import shutil
import subprocess
import threading
import time
import urllib.parse
import urllib.request
import uuid

BASE = Path(__file__).resolve().parent
REPO = BASE / 'Minw913-FrontierOR-9bd047c'
DATA = REPO / 'frontier-or'
SCOPE = BASE / 'scope.json'
API = 'https://huggingface.co/api/datasets/frontieror/FrontierOR'
HF = 'https://huggingface.co/datasets/frontieror/FrontierOR/resolve/'
MIB = 1024 ** 2
DESCRIPTORS = ('problem_description.txt', 'mathematical_formulation.md',
               'instance_schema.json', 'solution_schema.json',
               'feasibility_check.py', 'solution_logger.py')
THREAD_ENV = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS')
LOCK = threading.RLock()
MANIFEST = {}
ARGS = None
LAUNCH_COUNT = 0
REFERENCE_SCORES = None
TASK_DIRECTIONS = None
OFFICIAL_METRICS = None
MEMORY_ADMISSION = threading.Condition()
MEMORY_RESERVATIONS = 0
ADMISSION_LOCAL = threading.local()


def acquire_memory_admission(allow_adaptive=True):
    global MEMORY_RESERVATIONS
    configured = ARGS.memory_mib * MIB
    wait_started = time.monotonic()
    deadline = wait_started + ARGS.memory_wait_seconds
    with MEMORY_ADMISSION:
        while True:
            sample = machine_memory_snapshot()
            reduced_reservation = 0 < MEMORY_RESERVATIONS < configured
            cap = configured
            if allow_adaptive and not MEMORY_RESERVATIONS:
                cap = min(configured, max(0, sample['available_bytes'] - 2048 * MIB) // MIB * MIB)
            required = cap + MEMORY_RESERVATIONS + 2048 * MIB
            if not reduced_reservation and cap >= 1024 * MIB and sample['available_bytes'] >= required:
                MEMORY_RESERVATIONS += cap
                ADMISSION_LOCAL.reserved = cap
                return {'machine_memory': sample, 'reserve_bytes': 2048 * MIB,
                        'job_budget_bytes': cap, 'configured_job_budget_bytes': configured,
                        'budget_provenance':('measured available minus 2 GiB reserve; reduced caps serialize; no inferred solver requirement'
                            if allow_adaptive else 'configured fixed cap admitted against measured available plus 2 GiB reserve'),
                        'other_reserved_bytes': MEMORY_RESERVATIONS - cap,
                        'admitted_evaluator_concurrency_at_launch':1 + int(MEMORY_RESERVATIONS > cap),
                        'required_available_bytes': required,
                        'admission_wait_seconds': round(time.monotonic() - wait_started, 3),
                        'admission_wait_limit_seconds': ARGS.memory_wait_seconds}
            if not MEMORY_RESERVATIONS or ARGS.memory_wait_seconds:
                remaining = deadline - time.monotonic()
                if remaining > 0:
                    MEMORY_ADMISSION.wait(timeout=min(5, remaining))
                    continue
                ADMISSION_LOCAL.denied = {'machine_memory':sample,
                    'proposed_job_budget_bytes':cap, 'configured_job_budget_bytes':configured,
                    'reserve_bytes':2048*MIB, 'required_available_bytes':required,
                    'minimum_adaptive_job_budget_bytes':1024*MIB,
                    'allow_adaptive':allow_adaptive, 'classification':'machine_memory_admission_deferred',
                    'admission_wait_seconds':round(time.monotonic() - wait_started, 3),
                    'admission_wait_limit_seconds':ARGS.memory_wait_seconds}
                return None
            MEMORY_ADMISSION.wait()


def run_official(tid, instance, digest):
    global MEMORY_RESERVATIONS
    ADMISSION_LOCAL.reserved = 0
    try:
        return _run_official(tid, instance, digest)
    finally:
        with MEMORY_ADMISSION:
            MEMORY_RESERVATIONS -= ADMISSION_LOCAL.reserved
            ADMISSION_LOCAL.reserved = 0
            MEMORY_ADMISSION.notify_all()


def comparison_counts(rows, population='large'):
    global REFERENCE_SCORES, TASK_DIRECTIONS, OFFICIAL_METRICS
    if not MANIFEST.get('metadata_provenance'):
        return {'comparison_state':'pending_pinned_reference_verification'}
    if REFERENCE_SCORES is None:
        with gzip.open(DATA/'metadata/gurobi_references.csv.gz','rt') as inp:
            REFERENCE_SCORES = {(r['task_id'],r['instance']):r for r in csv.DictReader(inp)}
        TASK_DIRECTIONS = {r['paper_id']:r['direction'] for r in json.loads((DATA/'metadata/paper_meta_info.json').read_text())}
        spec = importlib.util.spec_from_file_location('frontieror_official_metrics',REPO/'scripts/compute_benchmark_main_metrics.py')
        OFFICIAL_METRICS = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(OFFICIAL_METRICS)
    quality = faster = joint = unavailable = numerical_ties = 0
    for row in rows:
        if ((population=='large' and not row['instance'].startswith('large_'))
                or (population=='tiny' and row['instance']!='tiny')):
            continue
        ref = REFERENCE_SCORES.get((row['paper_id'],row['instance']))
        if row.get('feasible','').lower()!='true' or row.get('fail_reason')=='runtime_error':
            continue
        try:
            objective = float(row['obj'])
            baseline = float(ref['objective_value'])
            elapsed = float(row['time'])
            published_time = float(ref['runtime'])
            if not all(math.isfinite(x) for x in (objective,baseline,elapsed,published_time)) or min(elapsed,published_time)<0:
                raise ValueError('invalid_values')
        except (TypeError,ValueError,KeyError):
            unavailable += 1
            continue
        zero = OFFICIAL_METRICS.PROVEN_ZERO_OPTIMUM_TOLERANCE.get((row['paper_id'],row['instance']))
        tolerance = zero if baseline==0 and zero is not None else max(1e-6,abs(baseline)*1e-9)
        reported = Decimal(row['obj'])
        reported_reference = Decimal(ref['objective_value'])
        better = reported<=reported_reference if TASK_DIRECTIONS[row['paper_id']]=='min' else reported>=reported_reference
        numerical_ties += not better and abs(objective-baseline)<=tolerance
        fast = elapsed<published_time
        quality += better
        faster += fast
        joint += better and fast
    return {'comparison_state':'official_scored_objectives_compared_to_pinned_published_metadata',
            f'equal_or_better_{population}_instances':quality,f'strict_reported_objective_equal_or_better_{population}_instances':quality,
            f'worse_objective_within_numerical_tolerance_{population}_instances':numerical_ties,
            f'strictly_faster_feasible_{population}_instances':faster,
            f'joint_equal_or_better_and_strictly_faster_{population}_instances':joint,
            f'feasible_{population}_comparison_unavailable':unavailable,
            'objective_comparison_policy':'strict Decimal comparison of reported objectives; NO tolerance merged into equal-or-better wins',
            'numerical_tie_bucket_tolerance':'max(1e-6,abs(reference)*1e-9); official cell-specific proven-zero tolerance where applicable; separately labelled only',
            'speed_comparison_caveat':'candidate local wall-clock vs published reference runtime; cross-machine, not controlled performance evidence'}


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def machine_memory_snapshot():
    memory = psutil.virtual_memory()
    swap = psutil.swap_memory()
    return {'measured_at':now(), 'total_bytes':memory.total, 'available_bytes':memory.available,
            'used_bytes':memory.used, 'swap_used_bytes':swap.used,
            'attribution':'whole machine; not attributed to benchmark-owned processes'}


def request(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={
        'User-Agent': 'FrontierOR-bounded-orchestrator/1'}), timeout=90)


def native_thread_audit(directory):
    calls = []
    for path in sorted(directory.glob('*.py')):
        if path.name == 'solution_logger.py':
            continue
        try:
            tree = ast.parse(path.read_text())
        except SyntaxError:
            continue
        aliases = set()
        modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == 'scipy.optimize':
                aliases.update(n.asname or n.name for n in node.names if n.name in ('milp', 'linprog'))
            elif isinstance(node, ast.ImportFrom) and node.module == 'scipy':
                modules.update(n.asname or n.name for n in node.names if n.name == 'optimize')
            elif isinstance(node, ast.Import):
                modules.update(n.asname or n.name.split('.')[0] for n in node.names
                               if n.name in ('scipy', 'scipy.optimize'))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = node.func.id if isinstance(node.func, ast.Name) else (
                node.func.attr if isinstance(node.func, ast.Attribute) else None)
            base = node.func
            while isinstance(base, ast.Attribute):
                base = base.value
            if not (isinstance(node.func, ast.Name) and name in aliases
                    or isinstance(node.func, ast.Attribute) and name in ('milp', 'linprog')
                    and isinstance(base, ast.Name) and base.id in modules):
                continue
            options = next((k.value for k in node.keywords if k.arg == 'options'), None)
            explicit = isinstance(options, ast.Dict) and any(
                isinstance(k, ast.Constant) and k.value == 'threads'
                and isinstance(v, ast.Constant) and type(v.value) is int and v.value == 1
                for k, v in zip(options.keys, options.values))
            calls.append({'file':path.name, 'line':node.lineno, 'function':name,
                          'sha256':sha(path), 'explicit_native_threads_1':explicit})
    return {'version':1, 'calls':calls,
            'missing_native_limit':[c for c in calls if not c['explicit_native_threads_1']],
            'evidence_kind':'source configuration inference; not benchmark-run thread measurement',
            'measured_benchmark_native_thread_count':None}


def audit_run_threads(tid, record):
    if 'native_thread_configuration_audit' not in record:
        directory = Path(record.get('results_csv', '')).parent/'solvers'/tid/ARGS.model
        record['native_thread_configuration_audit'] = native_thread_audit(directory)
    record['requested_numerical_threads_per_solver'] = record.pop('numerical_threads_per_solver', 1)
    return record['native_thread_configuration_audit']


def save():
    with LOCK:
        MANIFEST['updated_at'] = now()
        MANIFEST['summary'] = {
            'tasks': len(MANIFEST['tasks']),
            'descriptors_ready': sum(t.get('descriptors_ready', False) for t in MANIFEST['tasks'].values()),
            'tiny_ready': sum(t.get('tiny_ready', False) for t in MANIFEST['tasks'].values()),
            'states': {s: sum(t.get('state') == s for t in MANIFEST['tasks'].values())
                       for s in sorted({t.get('state', 'pending') for t in MANIFEST['tasks'].values()})}}
        rows = []
        timing_eligible_rows = []
        affected_attempts = {}
        affected_cells = []
        for tid in MANIFEST['canonical_ids']:
            t = MANIFEST['tasks'][tid]
            preserved = t.get('preserved_official_results', {}).get('rows', [])
            rows.extend(preserved)
            timing_eligible_rows.extend(preserved)
            completed = {}
            for r in t.get('evaluation_history', []) + list(t.get('evaluations', {}).values()):
                audit = audit_run_threads(tid, r)
                if audit['missing_native_limit']:
                    affected_attempts[tid, r.get('results_csv')] = {
                        'task_id':tid, 'results_csv':r.get('results_csv'), 'solver_sha256':r.get('solver_sha256'),
                        'state':r.get('state'), 'sites':audit['missing_native_limit'],
                        'instance':(r.get('result') or {}).get('instance') or Path(r.get('results_csv','')).parent.parent.name}
                if r.get('state')=='complete' and r.get('result'):
                    completed[r['result']['instance']] = r
            for instance, r in completed.items():
                rows.append(r['result'])
                if r['native_thread_configuration_audit']['missing_native_limit']:
                    affected_cells.append({'task_id':tid, 'instance':instance, 'results_csv':r.get('results_csv')})
                else:
                    timing_eligible_rows.append(r['result'])
        if rows:
            csvpath = BASE / 'official-results-all180.csv'
            csvtemp = csvpath.with_suffix('.csv.tmp')
            with csvtemp.open('w', newline='') as out:
                writer = csv.DictWriter(out, fieldnames=list(rows[0]), extrasaction='ignore')
                writer.writeheader()
                writer.writerows(rows)
            os.replace(csvtemp, csvpath)
            MANIFEST['official_results_csv'] = str(csvpath)
        bycell = {(r['paper_id'], r['instance']): r for r in rows}
        def feasible(r):
            return str(r.get('feasible', '')).lower() == 'true'
        def executed(r):
            return r.get('fail_reason') != 'runtime_error' and r.get('status') not in ('skip', '', None)
        large_rows = [r for r in rows if r['instance'].startswith('large_')]
        authored = [tid for tid in MANIFEST['canonical_ids'] if solver_path(tid).is_file()]
        missing = {}
        for tid in MANIFEST['canonical_ids']:
            missing[tid] = [instance for instance in ['tiny']+['large_'+str(i) for i in range(1,6)]
                            if (tid,instance) not in bycell]
            MANIFEST['tasks'][tid]['pending_instances'] = missing[tid]
        missing = {tid:instances for tid,instances in missing.items() if instances}
        summary = MANIFEST['summary']
        summary.update(denominators={'tasks':180,'tiny_instances':180,'large_instances':900,'new_task_ownership':179},
            tasks_with_authored_solver=len(authored),
            official_tiny_attempted=sum((tid,'tiny') in bycell for tid in MANIFEST['canonical_ids']),
            latest_official_tiny_completed=sum(t.get('evaluations',{}).get('tiny',{}).get('state')=='complete'
                or any(r.get('instance')=='tiny' for r in t.get('preserved_official_results',{}).get('rows',[]))
                for t in MANIFEST['tasks'].values()),
            official_tiny_running_tasks=[tid for tid,t in MANIFEST['tasks'].items()
                                        if t.get('evaluations',{}).get('tiny',{}).get('state')=='running'],
            aggregate_row_policy='latest completed official row per task/instance; history retained during retries',
            official_tiny_pass=sum(r['instance']=='tiny' and r['status']=='pass' for r in rows),
            raw_checker_tiny_feasible=sum(r['instance']=='tiny' and feasible(r) for r in rows),
            runtime_success_tiny_feasible=sum(r['instance']=='tiny' and feasible(r) and executed(r) for r in rows),
            official_large_attempted=len(large_rows),
            large_cells_with_launch_record=sum(instance.startswith('large_')
                for t in MANIFEST['tasks'].values() for instance,r in t.get('evaluations',{}).items()
                if r.get('command')) + sum(r['instance'].startswith('large_')
                for t in MANIFEST['tasks'].values() for r in t.get('preserved_official_results',{}).get('rows',[])),
            official_large_running_cells=[{'task_id':tid,'instance':instance}
                for tid,t in MANIFEST['tasks'].items() for instance,r in t.get('evaluations',{}).items()
                if instance.startswith('large_') and r.get('state')=='running'],
            official_large_completed_rows=len(large_rows),
            official_large_pass=sum(r.get('status')=='pass' for r in large_rows),
            tasks_all_five_large_pass=sum(all((tid,'large_'+str(i)) in bycell
                and bycell[tid,'large_'+str(i)].get('status')=='pass' for i in range(1,6))
                for tid in MANIFEST['canonical_ids']),
            raw_checker_large_feasible=sum(feasible(r) for r in large_rows),
            runtime_success_large_feasible=sum(feasible(r) and executed(r) for r in large_rows),
            tasks_any_large_feasible=sum(any(feasible(r) and executed(r) for r in large_rows if r['paper_id']==tid)
                                         for tid in MANIFEST['canonical_ids']),
            tasks_all_five_large_feasible=sum(all((tid,'large_'+str(i)) in bycell
                and feasible(bycell[tid,'large_'+str(i)]) and executed(bycell[tid,'large_'+str(i)]) for i in range(1,6))
                for tid in MANIFEST['canonical_ids']),
            tasks_pending=len(missing), pending_instance_count=sum(map(len,missing.values())),
            resource_failure_instances=[{'task_id':tid,'instance':instance,'peak_rss_bytes':r.get('peak_process_tree_rss_bytes')}
                for tid,t in MANIFEST['tasks'].items() for instance,r in t.get('evaluations',{}).items()
                if r.get('state')=='memory_exceeded'],
            infrastructure_error_tasks=[tid for tid,t in MANIFEST['tasks'].items()
                if any(r.get('state') in ('harness_error','checker_dependency_blocked','interrupted_infrastructure')
                       or r.get('infrastructure_diagnosis') for r in t.get('evaluations',{}).values())],
            solver_runtime_error_tasks=[tid for tid,t in MANIFEST['tasks'].items()
                if any((r.get('result') or {}).get('fail_reason')=='runtime_error' for r in t.get('evaluations',{}).values())])
        current_source = []
        stale_source = []
        MANIFEST['source_inventory_at'] = now()
        for tid,t in MANIFEST['tasks'].items():
            current = solver_path(tid)
            t['current_solver_sha256'] = sha(current) if current.is_file() else None
            r = t.get('evaluations',{}).get('tiny',{})
            if r.get('state')=='complete' and r.get('result'):
                helpers = [{'name':p.name,'sha256':sha(p)} for p in sorted(current.parent.glob('*.py'))
                           if p.name not in ('code.py','solution_logger.py')]
                frozen_helpers = r.get('execution_contract',{}).get('authored_helpers',[])
                if r.get('solver_sha256')==t.get('current_solver_sha256') and helpers==frozen_helpers:
                    current_source.append(tid)
                else:
                    stale_source.append(tid)
        summary.update(official_tiny_matches_current_source_inventory=len(current_source),
            official_tiny_pass_matches_current_source_inventory=sum(MANIFEST['tasks'][tid]['evaluations']['tiny']['result']['status']=='pass'
                                                                    for tid in current_source),
            stale_official_tiny_source_tasks=stale_source,
            source_inventory_at=MANIFEST.get('source_inventory_at'))
        current_large = []
        stale_large = []
        for tid,t in MANIFEST['tasks'].items():
            current = solver_path(tid)
            helpers = [{'name':p.name,'sha256':sha(p)} for p in sorted(current.parent.glob('*.py'))
                       if p.name not in ('code.py','solution_logger.py')]
            for instance,r in t.get('evaluations',{}).items():
                if not instance.startswith('large_') or r.get('state')!='complete' or not r.get('result'):
                    continue
                if (r.get('solver_sha256')==t.get('current_solver_sha256')
                        and r.get('execution_contract',{}).get('authored_helpers',[])==helpers):
                    current_large.append(r['result'])
                else:
                    stale_large.append({'task_id':tid,'instance':instance})
        summary.update(new_current_source_large_completed=len(current_large),
            new_current_source_large_pass=sum(r.get('status')=='pass' for r in current_large),
            new_current_source_large_runtime_success_feasible=sum(feasible(r) and executed(r) for r in current_large),
            stale_large_snapshot_cells=stale_large)
        summary['denominators']['all_instances'] = 1080
        summary.update(official_all_completed_rows=len(rows),
            official_all_pass=sum(r.get('status')=='pass' for r in rows),
            raw_checker_all_feasible=sum(feasible(r) for r in rows),
            runtime_success_all_feasible=sum(feasible(r) and executed(r) for r in rows))
        for population in ('large','tiny','all'):
            summary.update(comparison_counts(rows, population))
        timing_counts = comparison_counts(timing_eligible_rows)
        for population in ('tiny','all'):
            counts = comparison_counts(timing_eligible_rows, population)
            summary[f'single_thread_timing_qualified_cross_machine_faster_{population}_instances'] = counts.get(f'strictly_faster_feasible_{population}_instances',0)
            summary[f'single_thread_timing_qualified_joint_quality_and_faster_{population}_instances'] = counts.get(f'joint_equal_or_better_and_strictly_faster_{population}_instances',0)
        summary.update(native_thread_limit_missing_attempt_count=len(affected_attempts),
            native_thread_limit_missing_task_count=len({r['task_id'] for r in affected_attempts.values()}),
            native_thread_limit_missing_completed_cells=affected_cells,
            native_thread_limit_missing_attempts=list(affected_attempts.values()),
            single_thread_timing_qualified_cross_machine_faster_large_instances=timing_counts.get('strictly_faster_feasible_large_instances',0),
            single_thread_timing_qualified_joint_quality_and_faster_large_instances=timing_counts.get('joint_equal_or_better_and_strictly_faster_large_instances',0),
            single_thread_timing_qualification='exclude snapshots missing explicit SciPy native control; source-derived, not measured benchmark pool size; cross-machine timing',
            objective_quality_thread_policy='all actual official objective/feasibility grades and strict quality wins retained regardless of thread configuration')
        MANIFEST['pending_instances'] = missing
        MANIFEST['authoring_provenance'] = {'new_solver_tasks':179,'authoring_backend':'inherited-agent',
            'authoring_model':'sol-6.1','evaluation_label':ARGS.model,
            'evaluation_label_role':'official harness compatibility alias, not measured GPT generation',
            'paid_generation_calls':False,'preserved_wangy_provenance':'original CSV rows and SHA256, separate from new authoring'}
        temp = SCOPE.with_suffix('.json.tmp')
        temp.write_text(json.dumps(MANIFEST, indent=2, sort_keys=True) + '\n')
        os.replace(temp, SCOPE)


def log(task, text):
    directory = BASE / 'orchestration-logs'
    directory.mkdir(exist_ok=True)
    with (directory / (task + '.log')).open('a') as out:
        out.write(now() + ' ' + text + '\n')


def classify(path):
    p = Path(path)
    if len(p.parts) == 3 and p.name in DESCRIPTORS:
        return 'descriptor'
    if len(p.parts) == 3 and p.name in ('solverless_witness.py','release_contract.py'):
        return 'checker_helper'
    if len(p.parts) == 4 and p.parts[2]=='solverless_models' and p.suffix=='.npz':
        return 'checker_coefficient_model'
    if len(p.parts) == 4 and p.parts[2] == 'instance':
        if p.name == 'tiny_instance.json':
            return 'tiny'
        if re.fullmatch(r'large_instance_[1-5]\.json', p.name):
            return 'large'
    return 'excluded_reference_or_unneeded'


def inventory():
    meta = json.loads((DATA / 'metadata/paper_meta_info.json').read_text())
    ids = sorted(row['paper_id'] for row in meta)
    if len(ids) != 180 or len(set(ids)) != 180:
        raise RuntimeError('Canonical metadata must contain exactly 180 unique paper IDs')
    old = MANIFEST.get('tasks', {})
    revision = MANIFEST.get('revision') or json.load(request(API))['sha']
    url = API + '/tree/' + revision + '?recursive=true&limit=1000'
    files = []
    while url:
        with request(url) as response:
            files.extend(x for x in json.load(response) if x['type'] == 'file')
            link = response.headers.get('Link', '')
        matches = re.findall(r'<([^>]+)>;\s*rel="?next"?', link)
        url = matches[0] if matches else None
    tasks = {}
    for tid in ids:
        t = old.get(tid, {'state': 'inventoried', 'blockers': [], 'evaluations': {}, 'downloads': {}})
        t['files'] = [{
            'path': x['path'], 'bytes': x.get('lfs', {}).get('size', x['size']),
            'sha256': x.get('lfs', {}).get('oid'), 'git_blob': x['oid'],
            'kind': classify(x['path'])}
            for x in files if x['path'].startswith('tasks/' + tid + '/')]
        tasks[tid] = t
    remote_ids = sorted({Path(x['path']).parts[1] for x in files if x['path'].startswith('tasks/')})
    missing = sorted(set(ids) - set(remote_ids))
    if missing:
        raise RuntimeError('Canonical IDs absent remotely: ' + ','.join(missing))
    MANIFEST.update(schema_version=1, dataset='frontieror/FrontierOR', revision=revision,
                    canonical_ids=ids, tasks=tasks, remote_noncanonical_ids=sorted(set(remote_ids)-set(ids)),
                    policy={'max_file_bytes': ARGS.max_file_mib*MIB, 'initial_file_size_policy_bytes':256*MIB, 'tiny_max_bytes': ARGS.tiny_mib*MIB,
                            'total_new_download_bytes': ARGS.budget_mib*MIB,
                            'workers': ARGS.workers, 'threads_per_worker': 1,
                            'memory_budget_per_evaluation_bytes': ARGS.memory_mib*MIB,
                            'large_gate': 'official tiny feasible=true and gap<=0.10; same solver SHA256',
                            'reference_assignment_downloads': False,
                            'scoring': 'official one_shot_eval.py only; consolidated references scoring-only',
                            'paid_calls': False},
                    remote_bytes=sum(x.get('lfs', {}).get('size', x['size']) for x in files),
                    inventory_at=now())
    save()
    grouped = [x for x in ids if x != 'wangy2020']
    for i in range(6):
        path = BASE / ('groups-' + str(i) + '.json')
        if not path.exists():
            path.write_text(json.dumps(grouped[i*30:(i+1)*30], indent=2) + '\n')
    print(json.dumps({'canonical_tasks': len(ids), 'remote_files': len(files), 'revision': revision}), flush=True)


def verify_metadata():
    expected = MANIFEST.get('metadata_remote_files')
    if not expected:
        url = API+'/tree/'+MANIFEST['revision']+'/metadata?recursive=true&limit=1000'
        with request(url) as response:
            expected = [x for x in json.load(response) if x['type']=='file']
        MANIFEST['metadata_remote_files'] = expected
    provenance = {}
    blockers = []
    for name in ('paper_meta_info.json','gurobi_references.parquet','gurobi_references.csv.gz'):
        path = DATA/'metadata'/name
        matches = [x for x in expected if x['path']=='metadata/'+name]
        if not path.is_file() or not matches:
            blockers.append('missing_metadata:'+name)
            continue
        content = path.read_bytes()
        remote = matches[0]
        digest = hashlib.sha256(content).hexdigest()
        expected_digest = remote.get('lfs',{}).get('oid')
        actual_blob = hashlib.sha1(b'blob '+str(len(content)).encode()+b'\0'+content).hexdigest()
        ok = digest==expected_digest if expected_digest else actual_blob==remote['oid']
        provenance[name] = {'local_path':str(path),'sha256':digest,'bytes':len(content),
                            'pinned_revision':MANIFEST['revision'],'remote_path':remote['path'],
                            'matches_pinned_revision':ok,'use':'scoring_only' if name.startswith('gurobi') else 'canonical_task_inventory'}
        if not ok:
            blockers.append('metadata_revision_hash_mismatch:'+name)
    MANIFEST['metadata_provenance'] = provenance
    MANIFEST['metadata_blockers'] = blockers
    MANIFEST['official_harness_provenance'] = {'path':str(REPO/'one_shot_eval.py'),
        'sha256':sha(REPO/'one_shot_eval.py'),'source_archive':'Minw913-FrontierOR-9bd047c'}
    save()
    if blockers:
        raise RuntimeError(';'.join(blockers))


def existing_ok(f):
    dest = DATA / f['path']
    if not dest.is_file() or dest.stat().st_size != f['bytes']:
        return False
    if f['sha256'] and f['kind']=='tiny':
        return sha(dest)==f['sha256']
    if f['kind'] in ('descriptor','checker_helper'):
        content = dest.read_bytes()
        if f['sha256']:
            return hashlib.sha256(content).hexdigest() == f['sha256']
        return hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest() == f['git_blob']
    algorithm = 'sha256' if f['sha256'] else 'git_blob'
    if f.get(algorithm):
        stat = dest.stat()
        signature = [stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,stat.st_ctime_ns]
        record = MANIFEST['tasks'][Path(f['path']).parts[1]].setdefault('downloads',{}).setdefault(f['path'],{})
        if record.get('verified_stat')==signature and record.get(algorithm)==f[algorithm]:
            return True
        if algorithm=='sha256':
            digest = sha(dest)
        else:
            blob = hashlib.sha1(b'blob '+str(f['bytes']).encode()+b'\0')
            with dest.open('rb') as inp:
                for chunk in iter(lambda: inp.read(MIB), b''):
                    blob.update(chunk)
            digest = blob.hexdigest()
        if digest!=f[algorithm]:
            return False
        record.update({algorithm:digest,'verified_stat':signature})
    return True


def downloaded_bytes():
    return sum(d.get('charged_bytes', 0) for t in MANIFEST['tasks'].values()
               for d in t.get('downloads', {}).values())


def download(tid, f):
    dest = DATA / f['path']
    with LOCK:
        t = MANIFEST['tasks'][tid]
        record = t['downloads'].setdefault(f['path'], {})
        if dest.exists():
            if existing_ok(f):
                record.update(state='existing', bytes=f['bytes'])
                return None
            return 'existing_integrity_mismatch:' + f['path']
        cap = ARGS.tiny_mib*MIB if f['kind'] == 'tiny' else ARGS.max_file_mib*MIB
        if f['bytes'] > cap:
            record.update(state='blocked', reason='file_size_limit', bytes=f['bytes'])
            return 'file_size_limit:' + f['path'] + ':' + str(f['bytes'])
        charged = record.get('charged_bytes', 0)
        if downloaded_bytes() + max(0, f['bytes']-charged) > ARGS.budget_mib*MIB:
            record.update(state='blocked', reason='overall_download_budget')
            return 'overall_download_budget:' + f['path']
        record.update(state='downloading', charged_bytes=max(charged, f['bytes']))
        save()
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_name(dest.name + '.partial')
    offset = partial.stat().st_size if partial.exists() else 0
    url = HF + MANIFEST['revision'] + '/' + urllib.parse.quote(f['path']) + '?download=true'
    try:
        while offset < f['bytes']:
            end = min(f['bytes']-1, offset+(64*MIB if f['bytes']>256*MIB else MIB)-1)
            headers = {'User-Agent':'FrontierOR-bounded-orchestrator/1',
                       'Range':'bytes='+str(offset)+'-'+str(end)}
            with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=90) as response:
                if response.status==206:
                    expected = 'bytes '+str(offset)+'-'+str(end)+'/'+str(f['bytes'])
                    if response.headers.get('Content-Range')!=expected:
                        raise ValueError('remote_range_mismatch')
                    remaining = end-offset+1
                elif response.status==200:
                    offset = 0
                    remaining = f['bytes']
                else:
                    raise ValueError('unexpected_download_status:'+str(response.status))
                with partial.open('ab' if offset else 'wb') as out:
                    while remaining:
                        chunk = response.read(min(MIB,remaining))
                        if not chunk:
                            raise ValueError('download_range_truncated')
                        out.write(chunk)
                        offset += len(chunk)
                        remaining -= len(chunk)
        if offset != f['bytes']:
            raise ValueError('download_size_mismatch:' + str(offset))
        h = hashlib.sha256()
        blob = hashlib.sha1(b'blob '+str(offset).encode()+b'\0')
        with partial.open('rb') as inp:
            for chunk in iter(lambda: inp.read(MIB), b''):
                h.update(chunk)
                blob.update(chunk)
        if f['sha256'] and h.hexdigest() != f['sha256']:
            raise ValueError('sha256_mismatch')
        if not f['sha256'] and f.get('git_blob') and blob.hexdigest()!=f['git_blob']:
            raise ValueError('git_blob_mismatch')
        # Hard link publishes without ever replacing an existing artifact.
        os.link(partial, dest)
        with LOCK:
            record.update(state='downloaded', bytes=offset, sha256=h.hexdigest(), completed_at=now())
        log(tid, 'downloaded ' + f['path'] + ' bytes=' + str(offset))
        return None
    except Exception as exc:
        reason = type(exc).__name__ + ':' + str(exc).split('https://')[0][:240]
        with LOCK:
            record.update(state='download_error', reason=reason)
        log(tid, 'download_error ' + f['path'] + ' ' + reason)
        return 'download_error:' + f['path'] + ':' + reason


def prepare(tid):
    t = MANIFEST['tasks'][tid]
    blockers = []
    for name in DESCRIPTORS:
        found = [f for f in t['files'] if f['kind']=='descriptor' and Path(f['path']).name==name]
        if not found:
            blockers.append('missing_remote_descriptor:' + name)
        for f in found:
            if ARGS.external_descriptors:
                err = None if existing_ok(f) else 'descriptor_owned_by_author:' + f['path']
            else:
                err = download(tid, f)
            if err and name != 'solution_logger.py':
                blockers.append(err)
    descriptor_errors = list(blockers)
    tiny = [f for f in t['files'] if f['kind'] == 'tiny']
    if not tiny:
        blockers.append('missing_remote_tiny_instance')
    for f in tiny:
        err = download(tid, f)
        if err:
            blockers.append(err)
    with LOCK:
        t.update(descriptors_ready=not descriptor_errors,
                 tiny_ready=bool(tiny) and all(existing_ok(f) for f in tiny),
                 blockers=blockers, state='ready_for_solver' if not blockers else 'download_blocked')
        save()
    return tid


def solver_path(tid):
    return BASE / 'solvers' / tid / ARGS.model / 'code.py'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_official(tid, instance, digest):
    global LAUNCH_COUNT
    t = MANIFEST['tasks'][tid]
    coefficient_sha = next((f.get('sha256') for f in t['files']
        if f['path'].endswith('/instance/'+('tiny_instance.json' if instance=='tiny' else 'large_instance_'+instance.split('_')[1]+'.json'))), None)
    coefficient_inputs = [{'path':f['path'],'sha256':f['sha256']} for f in t['files']
        if classify(f['path'])=='checker_coefficient_model' and Path(f['path']).stem==coefficient_sha
        and tid in ('cote2018','liao2020','rahmaniani2018','rahmaniani2022')]
    helper_inputs = [{'name':p.name,'sha256':sha(p)} for p in sorted(solver_path(tid).parent.glob('*.py'))
                     if p.name not in ('code.py','solution_logger.py')]
    contract = {'code_sha256':digest,'declared_coefficient_inputs':coefficient_inputs,
                'authored_helpers':helper_inputs,'dataset_revision':MANIFEST['revision'],
                'official_harness_sha256':sha(REPO/'one_shot_eval.py'),
                'official_execution_sha256':sha(REPO/'trusted_eval_infra/execution.py'),
                'official_logger_sha256':sha(REPO/'scripts/utils/solution_logger.py'),
                'orchestration_protocol':3}
    if instance!='tiny':
        gate = t.get('evaluations',{}).get('tiny',{})
        gate_contract = gate.get('execution_contract',{})
        gate_keys = ('code_sha256','authored_helpers','dataset_revision','official_harness_sha256',
                     'official_execution_sha256','official_logger_sha256')
        if not tiny_passed(gate,digest) or any(gate_contract.get(k)!=contract[k] for k in gate_keys):
            with LOCK:
                t.setdefault('source_change_before_launch_deferrals',{})[instance] = {
                    'recorded_at':now(), 'classification':'tiny gate contract changed before launch; no official evaluator launched',
                    'tiny_gate_contract':gate_contract, 'proposed_execution_contract':contract}
                save()
            return {'state':'deferred_source_changed_before_launch','solver_sha256':digest,
                    'reason':'source_helper_or_harness_changed_since_tiny_gate:rerun_tiny_first'}
    fingerprint = hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    previous = t['evaluations'].get(instance)
    if previous and not previous.get('input_fingerprint') and not coefficient_inputs:
        oldroot = Path(previous.get('results_csv','')).parent
        oldsource = oldroot/'solvers'/tid/ARGS.model/'code.py'
        oldharness = oldroot/'official/one_shot_eval.py'
        oldlogger = oldsource.parent/'solution_logger.py'
        oldhelpers = [{'name':p.name,'sha256':sha(p)} for p in sorted(oldsource.parent.glob('*.py'))
                      if p.name not in ('code.py','solution_logger.py')]
        if (oldsource.is_file() and oldharness.is_file() and oldlogger.is_file()
                and sha(oldsource)==digest and sha(oldharness)==contract['official_harness_sha256']
                and sha(oldlogger)==contract['official_logger_sha256'] and oldhelpers==helper_inputs):
            previous.update(input_fingerprint=fingerprint,execution_contract=contract,
                fingerprint_migration='verified immutable source/helpers/logger/harness snapshots; same pinned dataset and unchanged scoring metadata')
    if (previous and previous.get('solver_sha256') == digest and previous.get('input_fingerprint')==fingerprint and previous.get('state') == 'complete'
            and not (instance=='tiny' and ARGS.fresh_tiny)
            and not previous.get('retry_after_infrastructure_fix')
            and (previous.get('result') or {}).get('fail_reason') not in ('checker_error','checker_unavailable','missing_instance')):
        return previous
    payload_name = 'tiny_instance.json' if instance=='tiny' else 'large_instance_'+instance.split('_')[1]+'.json'
    payload_bytes = next(f['bytes'] for f in t['files'] if f['path'].endswith('/instance/'+payload_name))
    admission = acquire_memory_admission(allow_adaptive=not ARGS.fixed_memory and payload_bytes <= 256*MIB)
    if admission is None:
        with LOCK:
            t.setdefault('machine_memory_admission_deferrals',{})[instance] = ADMISSION_LOCAL.denied
            save()
        return {'state':'deferred_machine_memory_headroom','solver_sha256':digest,
                'reason':'available machine memory below admissible job cap plus 2 GiB reserve'}
    if (previous and previous.get('state')=='memory_exceeded'
            and previous.get('input_fingerprint')==fingerprint
            and admission['job_budget_bytes']<=previous.get('memory_budget_bytes',0)):
        return previous
    with LOCK:
        if LAUNCH_COUNT >= ARGS.max_launches:
            return {'state':'deferred_batch_limit','solver_sha256':digest}
        LAUNCH_COUNT += 1
    runid = dt.datetime.now().strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:8]
    run = BASE / 'orchestration-runs' / tid / instance / runid
    run.mkdir(parents=True)
    runtime = run / 'official'
    runtime.mkdir()
    for p in REPO.iterdir():
        if p.name in ('eval', 'frontier-or'):
            continue
        if p.name == 'one_shot_eval.py':
            shutil.copy2(p, runtime / p.name)
        else:
            (runtime / p.name).symlink_to(p, target_is_directory=p.is_dir())
    official_copy = runtime/'one_shot_eval.py'
    original_source = official_copy.read_text()
    diagnostic_tee = '''_frontieror_original_run_generated_code = run_generated_code

def _frontieror_diagnostic_run_generated_code(*args, **kwargs):
    result = _frontieror_original_run_generated_code(*args, **kwargs)
    with open(os.environ["FRONTIEROR_NATIVE_OUTPUT_LOG"], "a", encoding="utf-8") as native:
        native.write("\\n=== native execution output ===\\n")
        native.write(str(result[1]))
        native.write("\\n")
    return result

run_generated_code = _frontieror_diagnostic_run_generated_code

'''
    anchor = 'if __name__ == "__main__":'
    if original_source.count(anchor)!=1:
        raise RuntimeError('Official diagnostic instrumentation anchor changed')
    instrumented_source = original_source.replace(anchor,diagnostic_tee+anchor)
    (runtime/'one_shot_eval.original.py').write_text(original_source)
    official_copy.write_text(instrumented_source)
    patch = run/'official-harness-diagnostics.patch'
    patch.write_text(''.join(difflib.unified_diff(original_source.splitlines(True),instrumented_source.splitlines(True),
                         fromfile='official-original',tofile='official-readonly-output-tee')))
    results = run / 'official-results.csv'
    snapshot_root = run / 'solvers'
    snapshot = snapshot_root / tid / ARGS.model / 'code.py'
    snapshot.parent.mkdir(parents=True)
    shutil.copy2(solver_path(tid), snapshot)
    authored_helpers = []
    for helper in solver_path(tid).parent.glob('*.py'):
        if helper.name in ('code.py','solution_logger.py'):
            continue
        shutil.copy2(helper, snapshot.parent/helper.name)
        authored_helpers.append({'name':helper.name,'sha256':sha(snapshot.parent/helper.name),'source':str(helper)})
    snapshot_digest = sha(snapshot)
    snapshot_helpers = sorted(({'name':h['name'],'sha256':h['sha256']} for h in authored_helpers), key=lambda h:h['name'])
    if snapshot_digest != digest or snapshot_helpers != helper_inputs:
        with LOCK:
            t.setdefault('source_change_before_launch_deferrals',{})[instance] = {
                'recorded_at':now(), 'snapshot_directory':str(run),
                'expected_solver_sha256':digest, 'snapshot_solver_sha256':snapshot_digest,
                'expected_helpers':helper_inputs, 'snapshot_helpers':snapshot_helpers,
                'classification':'source_changed_before_launch; no official evaluator launched'}
            save()
        return {'state':'deferred_source_changed_before_launch','solver_sha256':digest}
    cmd = [str(BASE / '.venv/bin/python'), str(runtime / 'one_shot_eval.py'),
           '--paper_id', tid, '--models', ARGS.model, '--instances', instance,
           '--reuse-code', 'all', '--code-root', str(snapshot_root),
           '--exec-mode', 'bare', '--cpus', '1', '--memory', str(admission['job_budget_bytes']//MIB)+'M',
           '--time_limit', str(ARGS.time_limit), '--max_debug_retries', '0',
           '--max_correct_retries', '0', '--paper_workers', '1', '--model_workers', '1',
           '--instance_workers', '1', '--results_csv', str(results), '--gurobi-source', 'references']
    env = os.environ.copy()
    for key in list(env):
        if any(s in key.upper() for s in ('API_KEY', 'TOKEN', 'SECRET', 'PASSWORD')):
            env.pop(key)
    env.update({key: '1' for key in THREAD_ENV})
    env['FRONTIER_OR_DATA_DIR'] = str(DATA)
    env['FRONTIEROR_NATIVE_OUTPUT_LOG'] = str(run/'native-output.log')
    record = {'state': 'running', 'started_at': now(), 'solver_sha256': digest,
              'log': str(run/'harness.log'), 'results_csv': str(results), 'command': cmd, 'authored_helpers':authored_helpers,
              'input_fingerprint':fingerprint,'execution_contract':contract,
              'native_output_log':str(run/'native-output.log'),
              'machine_memory_launch_admission':admission,
              'evaluator_concurrency_workers':ARGS.workers, 'requested_numerical_threads_per_solver':1,
              'timing_provenance':'local wall-clock under configured evaluator concurrency; published runtime is cross-machine, not controlled speedup',
              'diagnostic_instrumentation':{'purpose':'read-only native output tee; original return tuple untouched',
                  'patch':str(patch),'patch_sha256':sha(patch),'instrumented_harness_sha256':sha(official_copy),
                  'scoring_source_sha256':contract['official_harness_sha256']}}
    if previous and previous.get('state')=='memory_exceeded' and previous.get('input_fingerprint')==fingerprint:
        record['resource_retry_budget_provenance'] = {
            'previous_failed_budget_bytes':previous['memory_budget_bytes'],
            'new_budget_bytes':admission['job_budget_bytes'],
            'reason':'higher admitted budget within approved configured ceiling'}
    with LOCK:
        if previous:
            t.setdefault('evaluation_history', []).append(previous)
        t['evaluations'][instance] = record
        t['state'] = 'evaluating_' + instance
        save()
    instance_name = 'tiny_instance.json' if instance=='tiny' else 'large_instance_'+instance.split('_')[1]+'.json'
    payload = next(f for f in t['files'] if f['path'].endswith('/instance/'+instance_name))
    for f in t['files']:
        if f['kind']=='checker_helper' or (f['kind']=='checker_coefficient_model'
                and Path(f['path']).stem == payload.get('sha256')):
            err = download(tid, f)
            if err:
                record.update(state='checker_dependency_blocked', reason=err)
                save()
                return record
    if tid in ('cote2018','liao2020','rahmaniani2018','rahmaniani2022'):
        declared = []
        for f in t['files']:
            if f['kind']=='checker_coefficient_model' and Path(f['path']).stem==payload.get('sha256'):
                inputs = snapshot.parent / 'inputs'
                inputs.mkdir(exist_ok=True)
                target = inputs / Path(f['path']).name
                shutil.copy2(DATA/f['path'], target)
                if sha(target) != f['sha256']:
                    raise RuntimeError('allowed_coefficient_input_hash_mismatch')
                declared.append({'path':f['path'],'revision':MANIFEST['revision'],
                    'sha256':f['sha256'],'local_snapshot':str(target),
                    'contents':'public variable names and constraint/objective coefficients; no solution assignments',
                    'purpose':'intentional allowed solver model input; model loading/build/solve inside timed CLI'})
        record['allowed_solver_inputs'] = declared
        t['allowed_solver_inputs'] = declared
    cap = admission['job_budget_bytes']
    try:
        if sha(solver_path(tid)) != digest:
            raise RuntimeError('solver_changed_before_launch')
        with (run/'harness.log').open('x') as out:
            proc = subprocess.Popen(cmd, cwd=runtime, env=env, stdout=out,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            peak, memory_exceeded = supervise_owned_process(proc, cap, ARGS.terminate_on_memory)
            completed = subprocess.CompletedProcess(cmd, proc.returncode)
            record.update(peak_process_tree_rss_bytes=peak, memory_budget_bytes=cap, memory_budget_exceeded=memory_exceeded,
                          memory_enforcement='terminate_process_tree' if ARGS.terminate_on_memory else 'measurement_only')
        rows = []
        if results.exists():
            with results.open() as inp:
                rows = [r for r in csv.DictReader(inp) if r['paper_id']==tid and r['instance']==instance]
        row = rows[-1] if rows else None
        output = (run/'harness.log').read_text(errors='replace')
        if row and row.get('fail_reason')=='runtime_error' and 'Symbol not found:' in output and 'libhighs' in output:
            record.update(infrastructure_diagnosis='native_HiGHS_dylib_ABI_collision',retry_after_infrastructure_fix=True)
        record.update(state=('memory_exceeded' if record.get('memory_budget_exceeded') and ARGS.terminate_on_memory else
                             ('complete' if completed.returncode==0 and row else 'harness_error')),
                      returncode=completed.returncode, result=row, finished_at=now())
        if sha(solver_path(tid)) != digest:
            record.update(source_changed_after_snapshot=True)
    except Exception as exc:
        record.update(state='harness_error', reason=type(exc).__name__+':'+str(exc), finished_at=now())
    with LOCK:
        save()
    log(tid, 'official ' + instance + ' ' + record['state'])
    return record


def supervise_owned_process(proc, cap, terminate_on_memory):
    root = psutil.Process(proc.pid)
    peak = 0
    memory_exceeded = False
    while proc.poll() is None:
        try:
            processes = [root] + root.children(recursive=True)
            rss = sum(p.memory_info().rss for p in processes if p.is_running())
            peak = max(peak, rss)
            if rss > cap:
                memory_exceeded = True
                if terminate_on_memory:
                    for p in reversed(processes):
                        try:
                            p.terminate()
                        except psutil.NoSuchProcess:
                            pass
                    # Popen must reap its root to preserve the exit status.
                    _, alive = psutil.wait_procs(processes[1:], timeout=3)
                    for p in alive:
                        try:
                            p.kill()
                        except psutil.NoSuchProcess:
                            pass
                    try:
                        proc.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        try:
                            root.kill()
                        except psutil.NoSuchProcess:
                            pass
                        proc.wait()
                    break
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        try:
            proc.wait(timeout=0.25)
        except subprocess.TimeoutExpired:
            pass
    return peak, memory_exceeded


def tiny_passed(record, digest):
    row = record.get('result') or {}
    return (record.get('state')=='complete' and record.get('solver_sha256')==digest
            and row.get('feasible','').lower()=='true' and row.get('status')=='pass'
            and row.get('fail_reason')!='runtime_error')


def evaluate(tid):
    if ARGS.stage=='large' and LAUNCH_COUNT>=ARGS.max_launches:
        return
    prepare(tid)
    t = MANIFEST['tasks'][tid]
    blockers = []
    path = solver_path(tid)
    if not path.is_file():
        blockers.append('missing_solver:' + str(path))
    if not t.get('descriptors_ready') or not t.get('tiny_ready'):
        blockers.append('descriptors_or_tiny_not_ready')
    if not (DATA/'metadata/gurobi_references.parquet').exists():
        blockers.append('missing_scoring_reference_table')
    if blockers:
        with LOCK:
            t.update(state='evaluation_blocked', blockers=blockers)
            save()
        return
    for f in t['files']:
        f['kind'] = classify(f['path'])
    digest = sha(path)
    audit = native_thread_audit(path.parent)
    t['current_native_thread_configuration_audit'] = audit
    if audit['missing_native_limit']:
        with LOCK:
            t.update(state='evaluation_blocked', blockers=['native_solver_thread_limit_missing:' + json.dumps(audit['missing_native_limit'])])
            save()
        return
    if ARGS.stage=='large':
        saved = t['evaluations'].get('tiny',{})
        contract = saved.get('execution_contract',{})
        helpers = [{'name':p.name,'sha256':sha(p)} for p in sorted(path.parent.glob('*.py'))
                   if p.name not in ('code.py','solution_logger.py')]
        unchanged = (saved.get('solver_sha256')==digest and contract.get('dataset_revision')==MANIFEST['revision']
            and contract.get('official_harness_sha256')==sha(REPO/'one_shot_eval.py')
            and contract.get('official_execution_sha256')==sha(REPO/'trusted_eval_infra/execution.py')
            and contract.get('official_logger_sha256')==sha(REPO/'scripts/utils/solution_logger.py')
            and contract.get('authored_helpers')==helpers)
        if not unchanged:
            with LOCK:
                t.update(state='evaluation_blocked',blockers=['tiny_gate_source_helper_or_harness_contract_changed:rerun_tiny_first'])
                save()
            return
    tiny = (t['evaluations'].get('tiny', {}) if ARGS.stage=='large'
            else run_official(tid, 'tiny', digest))
    if tiny.get('state') in ('deferred_batch_limit','deferred_source_changed_before_launch','deferred_machine_memory_headroom'):
        with LOCK:
            t.update(state='pending_evaluation_batch', blockers=[tiny['state']+':tiny'])
            save()
        return
    if tiny.get('state') != 'complete':
        blockers.append('official_evaluator_infrastructure:'+tiny.get('state','unknown')+':'+str(tiny.get('reason',tiny.get('returncode'))))
    elif not tiny_passed(tiny, digest):
        row = tiny.get('result') or {}
        blockers.append('tiny_gate_failed:' + json.dumps({k:row.get(k) for k in ('status','fail_reason','feasible','gap','error')}))
    elif ARGS.stage != 'tiny':
        for i in [int(instance.split('_')[1]) for instance in (ARGS.instances or ['large_'+str(j) for j in range(1,6)])]:
            if LAUNCH_COUNT>=ARGS.max_launches:
                blockers.append('deferred_batch_limit:large_'+str(i))
                break
            instance = 'large_' + str(i)
            if any(e.get('instance') == instance and e.get('classification') == 'input_contract_conflict'
                   for e in t.get('contract_blocker_evidence', [])):
                blockers.append('input_contract_conflict:' + instance)
                continue
            match = [f for f in t['files'] if f['path'].endswith('/large_instance_'+str(i)+'.json') and f['kind']=='large']
            if not match:
                blockers.append('missing_remote_instance:' + instance)
                continue
            if ARGS.workers > 1 and match[0]['bytes'] > 256*MIB:
                blockers.append('requires_serial_huge_instance:' + instance + ':' + str(match[0]['bytes']))
                continue
            err = download(tid, match[0])
            if err:
                blockers.append(err)
                continue
            if sha(path) != digest:
                blockers.append('solver_changed_after_tiny_gate')
                break
            res = run_official(tid, instance, digest)
            if res['state'] in ('deferred_machine_memory_headroom','deferred_source_changed_before_launch'):
                blockers.append(res['state'] + ':' + instance)
            elif res['state'] != 'complete':
                blockers.append('official_evaluation_error:' + instance + ':' + res['state'])
    with LOCK:
        state = ('tiny_evaluated' if ARGS.stage=='tiny' else 'evaluated') if not blockers else (
            'pending_evaluation_batch' if all(b.startswith(('deferred_batch_limit:','deferred_machine_memory_headroom:','deferred_source_changed_before_launch:')) for b in blockers) else 'evaluation_blocked')
        t.update(state=state, blockers=blockers)
        save()


def evaluate_batches(selected):
    global LAUNCH_COUNT
    automatic = ARGS.stage=='large' and not ARGS.tasks
    while True:
        LAUNCH_COUNT = 0
        with futures.ThreadPoolExecutor(max_workers=ARGS.workers) as pool:
            list(pool.map(evaluate, selected))
        MANIFEST['evaluation_queue'].update(finished_at=now(),
            machine_memory_at_end=machine_memory_snapshot(), actual_launches=LAUNCH_COUNT)
        save()
        print(json.dumps(MANIFEST['summary'], indent=2), flush=True)
        if not automatic or LAUNCH_COUNT < ARGS.max_launches:
            MANIFEST['evaluation_queue']['continuation_exit'] = {
                'reason':'explicit_bounded_tasks' if not automatic else 'selected_pass_exhausted_below_batch_limit',
                'actual_launches':LAUNCH_COUNT,
                'pending_cells':[{
                    'task_id':tid, 'instance':instance,
                    'task_blockers':MANIFEST['tasks'][tid].get('blockers',[]),
                    'evaluation_state':MANIFEST['tasks'][tid].get('evaluations',{}).get(instance,{}).get('state'),
                    'memory_admission_deferral':MANIFEST['tasks'][tid].get('machine_memory_admission_deferrals',{}).get(instance)
                } for tid in selected for instance in MANIFEST.get('pending_instances',{}).get(tid,[])
                    if instance.startswith('large_')]}
            save()
            return
        selected = sorted(selected, key=lambda tid: sum(
            instance.startswith('large_') and bool(r.get('command'))
            for instance,r in MANIFEST['tasks'][tid].get('evaluations',{}).items()))
        queue = dict(MANIFEST['evaluation_queue'])
        queue.update(planned_at=now(), task_ids=selected,
            machine_memory_at_start=machine_memory_snapshot())
        for key in ('finished_at','machine_memory_at_end','actual_launches'):
            queue.pop(key,None)
        MANIFEST['evaluation_queue'] = queue
        MANIFEST.setdefault('evaluation_batch_history',[]).append(queue)
        save()


def main():
    global MANIFEST, ARGS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('inventory','download','evaluate','all','status'))
    parser.add_argument('--tasks', nargs='+')
    parser.add_argument('--instances', nargs='+', choices=['large_'+str(i) for i in range(1,6)],
                        help='Restrict large cells for targeted repaired-source replay')
    parser.add_argument('--external-descriptors', action='store_true', help='Authors own descriptor download; only check descriptor presence')
    parser.add_argument('--stage', choices=('tiny','large','all'), default='tiny')
    parser.add_argument('--max-launches', type=int, default=8, help='Bound harness launches per batch; unscoped large runs continue until eligible work is exhausted')
    parser.add_argument('--fresh-tiny', action='store_true', help='Require a genuine new official tiny validation for explicit tasks')
    parser.add_argument('--after-fresh-tiny-receipt', type=Path, help='Wait for an identified fresh tiny driver and require its actual PASS before continuing')
    parser.add_argument('--wait-lock', action='store_true', help='Queue after the active orchestrator')
    parser.add_argument('--terminate-on-memory', action='store_true', help='Requires approval: terminate only launched evaluator process trees over the RSS budget')
    parser.add_argument('--workers', type=int, choices=(1,2), default=2)
    parser.add_argument('--budget-mib', type=int, default=32768)
    parser.add_argument('--max-file-mib', type=int, default=256, help='Initial download policy; >256 requires serial workers and explicitly approved large stage')
    parser.add_argument('--large-approved', action='store_true', help='Requires root/user approval: enable large evaluations')
    parser.add_argument('--tiny-mib', type=int, default=16)
    parser.add_argument('--memory-mib', type=int, default=6144)
    parser.add_argument('--fixed-memory', action='store_true', help='Admit only the full configured cap plus 2 GiB reserve; never reduce the actual cap')
    parser.add_argument('--memory-wait-seconds', type=int, default=0, help='Bounded resident wait for measured memory admission; default immediate deferral')
    parser.add_argument('--time-limit', type=int, default=60)
    parser.add_argument('--model', default='gpt-5.3-codex')
    ARGS = parser.parse_args()
    if ARGS.memory_wait_seconds < 0:
        parser.error('--memory-wait-seconds must be nonnegative')
    if ARGS.fresh_tiny and (ARGS.stage!='tiny' or not ARGS.tasks):
        parser.error('--fresh-tiny requires explicit tasks and --stage tiny')
    if ARGS.instances and ARGS.stage=='tiny':
        parser.error('--instances restricts large cells; use --stage large or all')
    if ARGS.stage in ('large','all') and not ARGS.large_approved:
        parser.error('Large stage paused: explicit root/user approval and --large-approved required')
    if ARGS.max_file_mib>256 and (ARGS.workers!=1 or not ARGS.large_approved):
        parser.error('Elevated initial file-size policy requires serial workers and explicit large approval')
    if min(ARGS.budget_mib, ARGS.tiny_mib, ARGS.memory_mib, ARGS.time_limit, ARGS.max_launches, ARGS.max_file_mib)<=0:
        parser.error('budgets and limits must be positive')
    if ARGS.model in ('.','..') or not re.fullmatch(r'[A-Za-z0-9_.-]+', ARGS.model):
        parser.error('model must be a simple directory name')
    if ARGS.action=='status':
        current = json.loads(SCOPE.read_text()) if SCOPE.exists() else {}
        print(json.dumps(current.get('summary',{}),indent=2))
        return
    validation_receipt = None
    if ARGS.after_fresh_tiny_receipt:
        if ARGS.stage!='large' or ARGS.tasks:
            parser.error('--after-fresh-tiny-receipt requires unscoped --stage large')
        validation_receipt = json.loads(ARGS.after_fresh_tiny_receipt.read_text())
        command = validation_receipt['command']
        if '--fresh-tiny' not in command or command[command.index('--stage')+1]!='tiny':
            parser.error('validation receipt must identify a genuine fresh tiny driver')
        try:
            validation_process = psutil.Process(validation_receipt['pid'])
            if validation_process.create_time()==validation_receipt['create_time']:
                validation_process.wait()
        except psutil.NoSuchProcess:
            pass
    with (BASE/'orchestration.lock').open('a') as lockfile:
        try:
            fcntl.flock(lockfile, fcntl.LOCK_EX | (0 if ARGS.wait_lock else fcntl.LOCK_NB))
        except BlockingIOError:
            parser.exit(2, 'Another orchestrator is running; read scope.json for live progress.\n')
        if SCOPE.exists():
            MANIFEST = json.loads(SCOPE.read_text())
        if ARGS.action=='inventory' or not MANIFEST.get('tasks'):
            inventory()
        if validation_receipt:
            command = validation_receipt['command']
            tid = command[command.index('--tasks')+1]
            gate = MANIFEST['tasks'][tid].get('evaluations',{}).get('tiny',{})
            csvpath = Path(gate.get('results_csv',''))
            started = dt.datetime.fromisoformat(validation_receipt['started_at']).timestamp()
            if (not tiny_passed(gate,sha(solver_path(tid)))
                    or (gate.get('result') or {}).get('status')!='pass'
                    or not csvpath.is_file() or csvpath.stat().st_mtime < started):
                parser.exit(2,'Fresh official tiny validation did not complete with matching PASS; no continuation launched\n')
        preserved = MANIFEST['tasks']['wangy2020']
        original = BASE / 'results-final.csv'
        if original.exists() and not preserved.get('preserved_official_results'):
            with original.open() as inp:
                rows = [r for r in csv.DictReader(inp) if r['paper_id']=='wangy2020']
            if len(rows)==6 and all(r['status']=='pass' for r in rows):
                preserved.update(state='preserved_evaluated',
                    preserved_official_results={'path': str(original), 'sha256': sha(original),
                                                'rows': rows, 'imported_at': now()},
                    blockers=[], descriptors_ready=True, tiny_ready=True)
        groups = [json.loads((BASE/('groups-'+str(i)+'.json')).read_text()) for i in range(6)]
        flat = sum(groups, [])
        if len(flat)!=179 or len(set(flat))!=179 or set(flat)|{'wangy2020'}!=set(MANIFEST['canonical_ids']):
            raise RuntimeError('Ownership must be 179 disjoint tasks plus preserved wangy2020')
        for i,g in enumerate(groups):
            for tid in g:
                MANIFEST['tasks'][tid]['owner'] = 'frontier-solvers-' + str(i)
        for tid,t in MANIFEST['tasks'].items():
            for f in t['files']:
                f['kind'] = classify(f['path'])
            source = solver_path(tid)
            t['current_solver_sha256'] = sha(source) if source.is_file() else None
        MANIFEST['source_inventory_at'] = now()
        verify_metadata()
        if ARGS.action in ('evaluate','all'):
            for process in psutil.process_iter(['pid','cmdline']):
                command = process.info.get('cmdline') or []
                if any(str(arg).startswith(str(BASE/'orchestration-runs')+'/')
                       and str(arg).endswith('/one_shot_eval.py') for arg in command):
                    try:
                        process.wait(timeout=360)
                    except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                        if process.is_running():
                            raise RuntimeError('Prior owned evaluator still running; no overlapping launch')
            for tid,t in MANIFEST['tasks'].items():
                for instance,record in t.get('evaluations',{}).items():
                    results = Path(record.get('results_csv',''))
                    if record.get('state')=='running' and results.is_file():
                        with results.open() as inp:
                            rows = [r for r in csv.DictReader(inp) if r.get('paper_id')==tid and r.get('instance')==instance]
                        if rows:
                            record.update(state='complete',result=rows[-1],recovered_after_orchestrator_interrupt=True)
                        else:
                            record.update(state='interrupted_infrastructure',reason='prior_orchestrator_interrupted_no_official_row')
                    elif record.get('state')=='running':
                        record.update(state='interrupted_infrastructure',reason='prior_orchestrator_interrupted_no_official_csv')
        selected = ARGS.tasks or [tid for tid in MANIFEST['canonical_ids'] if tid!='wangy2020']
        if ARGS.action in ('evaluate','all') and ARGS.stage=='tiny' and not ARGS.tasks:
            buckets = []
            for group in groups:
                buckets.append([tid for tid in group
                    if not MANIFEST['tasks'][tid].get('evaluations',{}).get('tiny',{}).get('result')])
            untested = [bucket[i] for i in range(max(map(len,buckets),default=0))
                        for bucket in buckets if i<len(bucket)]
            selected = untested + [tid for tid in selected if tid not in set(untested)]
        if ARGS.action in ('evaluate','all') and ARGS.stage=='large' and not ARGS.tasks:
            selected = [group[i] for i in range(max(map(len,groups))) for group in groups if i<len(group)]
            selected.sort(key=lambda tid: sum(instance.startswith('large_') and bool(r.get('command'))
                for instance,r in MANIFEST['tasks'][tid].get('evaluations',{}).items()))
        MANIFEST['evaluation_queue'] = {'stage':ARGS.stage,'task_ids':selected,'instances':ARGS.instances,'planned_at':now(),
            'policy':'explicit_tasks' if ARGS.tasks else ('balanced_owner_first_attempt_first' if ARGS.stage=='large' else 'untested_first_balanced_owners'),
            'evaluator_concurrency_workers':ARGS.workers, 'requested_numerical_threads_per_solver':1,
            'max_file_bytes':ARGS.max_file_mib*MIB, 'memory_budget_per_evaluation_bytes':ARGS.memory_mib*MIB,
            'machine_memory_at_start':machine_memory_snapshot()}
        MANIFEST.setdefault('evaluation_batch_history', []).append(MANIFEST['evaluation_queue'])
        unknown = sorted(set(selected)-set(MANIFEST['canonical_ids']))
        if unknown:
            parser.error('Unknown canonical tasks: ' + ','.join(unknown))
        MANIFEST['policy'].pop('memory_limit_per_evaluation_bytes',None)
        MANIFEST['policy'].update(max_file_bytes=ARGS.max_file_mib*MIB,large_stage_approved=ARGS.large_approved,
                                  memory_enforcement='terminate_owned_tree_on_measured_RSS_exceed' if ARGS.terminate_on_memory else 'measurement_only_no_termination_approval',
                                  stage=ARGS.stage,
                                  total_new_download_bytes=ARGS.budget_mib*MIB,
                                  tiny_max_bytes=ARGS.tiny_mib*MIB,
                                  memory_budget_per_evaluation_bytes=ARGS.memory_mib*MIB,
                                  workers=ARGS.workers)
        if ARGS.action in ('download','all'):
            with futures.ThreadPoolExecutor(max_workers=ARGS.workers) as pool:
                for tid in pool.map(prepare, selected):
                    print('prepared ' + tid, flush=True)
        if ARGS.action in ('evaluate','all'):
            evaluate_batches(selected)
        else:
            MANIFEST['evaluation_queue'].update(finished_at=now(),machine_memory_at_end=machine_memory_snapshot())
            save()
            print(json.dumps(MANIFEST['summary'], indent=2), flush=True)


if __name__ == '__main__':
    main()
