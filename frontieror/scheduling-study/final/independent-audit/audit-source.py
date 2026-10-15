"""Read-only full audit after all180 final executions; writes a new evidence folder."""
import argparse
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import statistics
import xml.etree.ElementTree as ET
from campaign import ROOT, read, save
from comparator_timing_calibration import validate_snapshots
from development_report import load, digest, require
from lisa_adapter import read_instance


def frozen_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_execution(slot, row, measurement, config):
    require(all(row.get(k) == v for k, v in slot.items()), 'Execution differs from frozen slot')
    inside = '/work/runs/' + slot['slot']
    if slot['solver'] == 'egrasp':
        argv = ['./egrasp', inside+'/input.xml', inside+'/output.xml', 's', '290']
    else:
        argv = ['./frozen-v2', inside+'/input.txt', inside+'/output.json', '290', str(slot['seed']), '1']
    require(measurement['argv'] == argv, 'Actual executable, paths, seed or internal budget differ')
    require(measurement['external_deadline_seconds'] == config['external_deadline_seconds'] == 300,
            'External deadline changed')


def text_instance(path):
    tokens = iter(map(int, path.read_text().split()))
    n, m = next(tokens), next(tokens)
    jobs = []
    for _ in range(n):
        job = dict(zip(('job_id','release_date','due_date','weight'), [next(tokens) for _ in range(4)]))
        job['operations'] = [dict(machine=next(tokens), processing_time=next(tokens)) for _ in range(m)]
        jobs.append(job)
    require(not list(tokens), 'Extra text input fields')
    return dict(num_jobs=n, num_machines=m, num_operations=n*m, jobs=jobs)


def independent_primary(records):
    values = []
    counts = {'candidate':0, 'comparator':0}
    for record in records:
        scores = {}
        for key in counts:
            scores[key] = []
            for row in record[key]:
                if row['status'] == 'VALID_IN_TIME':
                    counts[key] += 1
                    scores[key].append(row['objective'])
                else:
                    require(row['status']=='SOLVER_FAILURE' and row['objective'] is None, 'Unresolved execution')
                    scores[key].append(None)
        total = 0
        for a in scores['candidate']:
            for b in scores['comparator']:
                if a is None:
                    total -= int(b is not None)
                elif b is None:
                    total += 1
                elif a < b:
                    total += 1
                elif a > b:
                    total -= 1
        values.append(total/9)
    return statistics.mean(values), counts


def audit(current_path, output):
    state = load(current_path)
    require(state['state']=='complete' and state['completed']==state['planned']==180,
            'Final evaluation is not complete')
    source = Path(state['evidence'])
    result = load(source/'results.json')
    config = load(source/'plan-before-generation.json')
    require(result['state']=='complete' and result['config']==config, 'Final config mismatch')
    remote = read(state['experiment_id'])
    require(remote['self']['status']=='complete' and remote['self']['config']==config, 'Trackinizer config/status mismatch')
    require(json.loads(remote['self']['outcome'])==result, 'Trackinizer outcome differs')
    conclusion = read(state['belief_id'])
    require(state['experiment_id'] in json.dumps(conclusion.get('edges',{}))+json.dumps(conclusion.get('backlinks',{})), 'Missing conclusion link')
    for name, sha in config['source_sha256'].items():
        require(digest(source/'source'/name)==sha, 'Frozen source changed: '+name)
    for name in ('comparator_timing_calibration.py','lisa_adapter.py','audit_schedule.py'):
        require(digest(ROOT/'pilot'/name)==config['source_sha256'][name], 'Validator helper changed: '+name)
    require(digest(ROOT/'data/tasks/bierwirth2017/feasibility_check.py')==config['official_checker_sha256'], 'Official checker changed')
    protocol = frozen_module(source/'source/final_protocol.py', 'audited_final_protocol')
    generator = frozen_module(source/'source/synthetic_instances_v2.py', 'audited_final_generator')
    analysis_code = frozen_module(source/'source/comparator_analysis.py', 'audited_final_analysis')
    require(config['plan']==protocol.instance_plan(), 'Instance seed plan differs')
    schedule = protocol.execution_plan()
    require(config['execution_schedule']==schedule and len(schedule)==180, 'Execution schedule differs')
    require(config['bootstrap_seed']==protocol.BOOTSTRAP_SEED and config['bootstrap_resamples']==10000, 'Analysis settings differ')
    require(config['seeds_v2']==list(protocol.SOLVER_SEEDS) and config['order_seed']==protocol.ORDER_SEED, 'Seed settings differ')
    manifest = load(source/'instance-manifest-before-execution.json')
    require(manifest==result['manifest'] and len(manifest)==30, 'Manifest differs')
    require([{k:r[k] for k in ('name','tightness','weights','seed')} for r in manifest]==config['plan'], 'Manifest seed plan differs')
    registered = datetime.datetime.fromisoformat(remote['self']['created']).timestamp()
    inputs = {}
    hashes = []
    for entry in manifest:
        directory = source/'instances'/entry['name']
        for name, sha in entry['sha256'].items():
            require(digest(directory/name)==sha, 'Input hash differs')
            require((directory/name).stat().st_mtime >= registered, 'Input predates registration')
        instance = load(directory/'instance.json')
        require(instance==generator.generate(entry['seed'],entry['tightness'],entry['weights']), 'Input does not match registered generator and seed')
        require(instance==read_instance(ET.parse(directory/'input.xml').getroot())==text_instance(directory/'input.txt'), 'Input translations differ')
        inputs[entry['name']] = entry
        hashes.append(entry['sha256']['instance.json'])
    require(len(set(hashes))==30, 'Duplicate final instances')
    prior = read(config['development_experiment'])
    prior_manifest = json.loads(prior['self']['outcome'])['manifest']
    require(not set(hashes)&{r['sha256']['instance.json'] for r in prior_manifest}, 'Development instance reused')
    container = load(source/'container-inspect.json')[0]
    require(container['Image']==config['image'], 'Image differs')
    host = container['HostConfig']
    require(host['NanoCpus']==1_000_000_000 and host['Memory']==512*1024*1024 and host['NetworkMode']=='none', 'Resource limits differ')
    binary_hashes = {line.split()[1]:line.split()[0] for line in (source/'container-source-sha256.txt').read_text().splitlines()}
    require(all(binary_hashes[name]==sha for name,sha in config['solver_sha256'].items()), 'Binary/source hash differs')
    require(binary_hashes['timed_solver.py']==config['source_sha256']['timed_solver.py'], 'Container timer differs')
    rows = result['results']
    require(len(rows)==180, 'Missing executions')
    require(set(x.name for x in (source/'runs').iterdir() if x.is_dir())=={s['slot'] for s in schedule}, 'Missing or additional execution directory')
    output.mkdir(parents=True,exist_ok=False)
    shutil.copy2(Path(__file__),output/'audit-source.py')
    save(output/'experiment-readback.json',remote)
    save(output/'conclusion-readback.json',conclusion)
    verified=[]
    try:
        for index,(slot,row) in enumerate(zip(schedule,rows)):
            original=source/'runs'/slot['slot']
            require(load(original/'result.json')==row,'Per-run result differs')
            for name,sha in inputs[slot['instance']]['sha256'].items():
                require(digest(original/name)==sha,'Executed input differs')
            measurement=load(original/'timing/measurement.json')
            require(measurement==row['measurement'],'Measurement differs')
            check_execution(slot,row,measurement,config)
            require(measurement['started_unix'] >= (source/'instance-manifest-before-execution.json').stat().st_mtime, 'Execution preceded manifest freeze')
            for sample in measurement['samples']:
                if sample['parseable']:
                    require(digest(original/'timing'/sample['snapshot'])==sample['sha256'], 'Captured bytes changed')
                    require(sample['observed_seconds']<=300, 'Late captured bytes')
            target=output/slot['slot'];target.mkdir()
            shutil.copy2(original/'instance.json',target/'instance.json')
            shutil.copytree(original/'timing',target/'timing')
            checks=validate_snapshots(target,load(target/'instance.json'),slot['solver'],measurement)
            require(checks==row['checks'],'Fresh validators differ from stored checks')
            valid=[c for c in checks if c.get('valid') and c['observed_seconds']<=300]
            reported=re.findall(r'best solution with TWT value:\s*([\d.]+)',(original/'timing/stdout.txt').read_text()) if slot['solver']=='egrasp' else []
            require(not any(c.get('error') for c in checks),'Unresolved validation error')
            if valid:
                require(row['status']=='VALID_IN_TIME' and row['objective']==min(c['objective'] for c in valid),'Selected objective/status differs')
                if slot['solver']=='egrasp':
                    require(reported and float(reported[-1])==valid[-1]['objective']==row['reported'],'EGRASP report differs')
            else:
                require(row['status']=='SOLVER_FAILURE' and row['objective'] is None,'Failure was dropped or assigned an objective')
            commitment=load(original/'trackinizer-commitment.json')
            require(commitment['slot']==slot['slot'] and commitment['completed']==index+1,'Commitment order differs')
            require(commitment['result_sha256']==digest(original/'result.json'),'Result commitment differs')
            partial=dict(state='partial',config=config,manifest=manifest,results=rows[:index+1])
            expected_hash=hashlib.sha256((json.dumps(partial,indent=2,default=str)+'\n').encode()).hexdigest()
            require(commitment['cumulative_sha256']==expected_hash,'Cumulative commitment differs')
            readback=load(original/'trackinizer-readback.json')
            progress=json.loads(readback['self']['outcome'])
            require(progress['completed']==index+1 and progress['commitments'][-1]==commitment,'Per-attempt Trackinizer receipt differs')
            verified.append(dict(slot=slot['slot'],status=row['status'],objective=row['objective']))
        records=protocol.analysis_records(config['plan'],rows)
        computed=analysis_code.analyze(records,config['bootstrap_seed'],resamples=config['bootstrap_resamples'])
        require(computed==result['analysis']==load(source/'analysis.json'),'Frozen analysis recomputation differs')
        primary,counts=independent_primary(records)
        require(abs(primary-computed['means']['head_to_head'])<1e-12,'Independent primary calculation differs')
        require(counts['candidate']==computed['candidate_valid'] and counts['comparator']==computed['comparator_valid'],'Success counts differ')
        supported=counts['candidate']>=counts['comparator'] and computed['intervals']['head_to_head'][0]>0
        require(supported==computed['failure_aware_advantage_supported'],'Decision differs from registered threshold')
        summary=dict(state='verified',experiment_id=state['experiment_id'],executions=180,analysis=computed,scope=config['claim_scope'])
        save(output/'verified-executions.json',verified);save(output/'summary.json',summary)
        return summary
    except Exception as exc:
        save(output/'audit-failure.json',dict(error=repr(exc),verified=verified));raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('current',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();print(json.dumps(audit(args.current,args.output),indent=2))
