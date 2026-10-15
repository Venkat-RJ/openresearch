#!/usr/bin/env python3
"""Read-only campaign snapshot and independently recomputed report evidence.

Writes only beneath this report directory. Never imports or runs solver code.
Use --snapshot-dir for a later report, preserving this interim snapshot.
"""
import argparse, ast, collections, csv, datetime as dt, gzip, hashlib, io, json, math
from decimal import Decimal
from pathlib import Path
import subprocess, zipfile

BASE = Path('research-workspace/Personal_experiments/frontieror-attempt')
DATA = BASE/'Minw913-FrontierOR-9bd047c/frontier-or'
ROOT = Path(__file__).resolve().parent
def digest(b): return hashlib.sha256(b).hexdigest()
def dump(p, v): p.write_text(json.dumps(v, indent=2, sort_keys=True)+'\n')
def write_csv(p, rows):
    with p.open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def capture(dest):
    if dest.exists():
        if (dest/'summary.json').exists(): raise SystemExit('Snapshot already exists. Supply a new --snapshot-dir.')
        scope=json.loads((dest/'scope.json').read_text())
        rows=list(csv.DictReader((dest/'official-results-all180.csv').open()))
        source={t:{p.name:p.read_bytes() for p in sorted((dest/'source-at-capture'/t).glob('*.py'))} for t in scope['canonical_ids']}
        lines=(dest/'benchmark-processes.txt').read_text().splitlines()
        captured=dt.datetime.fromisoformat(lines.pop(0).removeprefix('Captured '))
        return analyze(dest,scope,rows,source,captured,lines)
    # Scope is written after its CSV. Verify both unchanged around source reads.
    for _ in range(12):
        sb=(BASE/'scope.json').read_bytes(); cb=(BASE/'official-results-all180.csv').read_bytes()
        scope=json.loads(sb); rows=list(csv.DictReader(io.StringIO(cb.decode())))
        source={}
        for tid in scope['canonical_ids']:
            folder=BASE/'solvers'/tid/'gpt-5.3-codex'
            source[tid]={p.name:p.read_bytes() for p in sorted(folder.glob('*.py'))}
        if sb!=(BASE/'scope.json').read_bytes() or cb!=(BASE/'official-results-all180.csv').read_bytes(): continue
        if len(rows)!=scope['summary']['official_all_completed_rows']: continue
        if any(digest(source[t]['code.py'])!=scope['tasks'][t].get('current_solver_sha256') for t in source): continue
        if any(b!= (BASE/'solvers'/t/'gpt-5.3-codex'/n).read_bytes() for t,files in source.items() for n,b in files.items()): continue
        break
    else: raise SystemExit('Could not capture consistent manifest, CSV and source. No snapshot written.')
    dest.mkdir(parents=True)
    (dest/'scope.json').write_bytes(sb); (dest/'official-results-all180.csv').write_bytes(cb)
    for name in ['gurobi_references.csv.gz','paper_meta_info.json']:
        b=(DATA/'metadata'/name).read_bytes(); assert digest(b)==scope['metadata_provenance'][name]['sha256']
        (dest/name).write_bytes(b)
    (dest/'orchestrate.py').write_bytes((BASE/'orchestrate.py').read_bytes())
    metrics=BASE/'Minw913-FrontierOR-9bd047c/scripts/compute_benchmark_main_metrics.py'
    (dest/'compute_benchmark_main_metrics.py').write_bytes(metrics.read_bytes())
    for tid,files in source.items():
        folder=dest/'source-at-capture'/tid;folder.mkdir(parents=True)
        for n,b in files.items(): (folder/n).write_bytes(b)
    captured=dt.datetime.now(dt.timezone.utc)
    proc=subprocess.check_output(['ps','-axo','pid,ppid,etime,%cpu,command'], text=True)
    lines=[l for l in proc.splitlines() if str(BASE) in l and 'capture_report.py' not in l]
    (dest/'benchmark-processes.txt').write_text('Captured '+captured.isoformat()+'\n'+'\n'.join(lines)+'\n')
    recent=sorted((BASE/'orchestration-logs').glob('*.log'),key=lambda p:p.stat().st_mtime,reverse=True)[:12]
    logs=[]
    for p in recent:
        logs.append({'source':str(p),'mtime_utc':dt.datetime.fromtimestamp(p.stat().st_mtime,dt.timezone.utc).isoformat(), 'tail':'\n'.join(p.read_text(errors='replace').splitlines()[-30:])})
    dump(dest/'recent-orchestration-logs.json',logs)
    return analyze(dest,scope,rows,source,captured,lines)

def analyze(dest,scope,rows,source,captured,lines):
    recs={}; mismatches=[]
    bycell={(r['paper_id'],r['instance']):r for r in rows}; assert len(bycell)==len(rows)
    for (tid,inst),row in bycell.items():
        t=scope['tasks'][tid]
        candidates=list(t.get('evaluation_history',[]))+list(t.get('evaluations',{}).values())
        matched=[e for e in candidates if e.get('result')==row]
        if matched: recs[(tid,inst)]=max(matched,key=lambda e:e.get('finished_at',''))
        elif row in t.get('preserved_official_results',{}).get('rows',[]):
            recs[(tid,inst)]={'preserved':True,'result':row,'preserved_provenance':t['preserved_official_results']}
        else: mismatches.append((tid,inst))
    assert not mismatches, mismatches
    dump(dest/'latest-row-records.json',{t+'/'+i:e for (t,i),e in recs.items()})
    refs={(r['task_id'],r['instance']):r for r in csv.DictReader(gzip.open(dest/'gurobi_references.csv.gz','rt'))}
    meta={r['paper_id']:r for r in json.loads((dest/'paper_meta_info.json').read_text())}
    zero={}
    for node in ast.parse((dest/'compute_benchmark_main_metrics.py').read_text()).body:
        if isinstance(node,ast.Assign) and any(isinstance(n,ast.Name) and n.id=='PROVEN_ZERO_OPTIMUM_TOLERANCE' for n in node.targets):
            assert {n.id for n in ast.walk(node.value) if isinstance(n,ast.Name)} <= {'range','i'}
            zero=eval(compile(ast.Expression(node.value),'zero-tolerance-only','eval'),{'__builtins__':{},'range':range})
    cells=[]; anomalies=[]
    for tid in scope['canonical_ids']:
        helpers=[{'name':n,'sha256':digest(b)} for n,b in source[tid].items() if n not in ('code.py','solution_logger.py')]
        for inst in ['tiny']+['large_'+str(i) for i in range(1,6)]:
            r=bycell.get((tid,inst),{}); e=recs.get((tid,inst),{}); ref=refs[(tid,inst)]
            current=bool(r) and not e.get('preserved') and e.get('solver_sha256')==digest(source[tid]['code.py']) and e.get('execution_contract',{}).get('authored_helpers',[])==helpers
            strict=bucket='unavailable';gain=pct=''; tol=''
            eligible=r.get('feasible','').lower()=='true' and r.get('fail_reason')!='runtime_error' and r.get('obj')
            if eligible:
                a,b=Decimal(r['obj']),Decimal(ref['objective_value'])
                assert a.is_finite() and b.is_finite()
                d=b-a if meta[tid]['direction']=='min' else a-b
                gain=str(d);pct=str(d/abs(b)*100) if b else ''
                tol=Decimal(str(zero.get((tid,inst),1e-6))) if b==0 else max(Decimal('0.000001'),abs(b)*Decimal('0.000000001'))
                strict='better' if d>0 else 'equal' if d==0 else 'worse'
                bucket='numerical_tie' if abs(d)<=tol else strict
            c={'task_id':tid,'instance':inst,'direction':meta[tid]['direction'],'recorded':bool(r),'status':r.get('status','pending'), 'feasible':r.get('feasible',''),'fail_reason':r.get('fail_reason',''), 'objective':r.get('obj',''), 'reference':ref['objective_value'], 'reference_status':ref['status'], 'strict_comparison':strict, 'tolerance_comparison':bucket,'comparison_tolerance':str(tol), 'signed_improvement':gain, 'improvement_percent':pct, 'source_current_at_capture':current, 'provenance_cohort':'preserved_pilot' if e.get('preserved') else 'current_source' if current else 'earlier_source' if r else 'pending','solver_sha256':e.get('solver_sha256',''), 'source_sha256_at_capture':digest(source[tid]['code.py']), 'runtime_seconds':r.get('time',''), 'published_runtime_seconds':ref['runtime'],'results_csv':e.get('results_csv',''), 'run_state':scope['tasks'][tid].get('evaluations',{}).get(inst,{}).get('state',''), 'error':r.get('error','')}
            cells.append(c)
            if strict=='better' and ref['status']=='optimal':anomalies.append(c)
    write_csv(dest/'case-comparisons.csv',cells);write_csv(dest/'optimal-reference-flags.csv',anomalies)
    taskrows=[]
    for tid in scope['canonical_ids']:
        tc=[c for c in cells if c['task_id']==tid];large=[c for c in tc if c['instance']!='tiny'];tiny=tc[0]
        taskrows.append({'task_id':tid,'direction':meta[tid]['direction'], 'tiny_status':tiny['status'], 'tiny_current_source':tiny['source_current_at_capture'], 'large_recorded':sum(c['recorded'] for c in large),'large_passes':sum(c['status']=='pass' for c in large), 'current_source_large_recorded':sum(c['source_current_at_capture'] for c in large),'current_source_large_passes':sum(c['source_current_at_capture'] and c['status']=='pass' for c in large), 'strict_improvements':sum(c['strict_comparison']=='better' for c in large), 'exact_ties':sum(c['strict_comparison']=='equal' for c in large),'improvements_beyond_tolerance':sum(c['tolerance_comparison']=='better' for c in large), 'missing_large_slots':sum(not c['recorded'] for c in large), 'all_five_pass':all(c['status']=='pass' for c in large), 'all_five_current_source_pass':all(c['status']=='pass' and c['source_current_at_capture'] for c in large),'state':scope['tasks'][tid]['state'], 'blockers':'; '.join(scope['tasks'][tid].get('blockers',[]))})
    write_csv(dest/'task-matrix-180.csv',taskrows)
    def population(cs):
        recorded=[c for c in cs if c['recorded']]
        return {'planned':len(cs),'recorded':len(recorded),'pass':sum(c['status']=='pass' for c in recorded),'failed':sum(c['status']!='pass' for c in recorded),'remaining':len(cs)-len(recorded), 'strict':dict(collections.Counter(c['strict_comparison'] for c in recorded)), 'tolerance':dict(collections.Counter(c['tolerance_comparison'] for c in recorded)), 'failures':dict(collections.Counter(c['fail_reason'] or c['status'] for c in recorded if c['status']!='pass'))}
    large=[c for c in cells if c['instance']!='tiny'];tiny=[c for c in cells if c['instance']=='tiny'];cur=[c for c in large if c['source_current_at_capture']]
    summary={'captured_utc':captured.isoformat(),'captured_ist':captured.astimezone(dt.timezone(dt.timedelta(hours=5,minutes=30))).isoformat(), 'manifest_updated_utc':scope['updated_at'],'dataset_revision':scope['revision'], 'large':population(large),'tiny':population(tiny),'all_five_pass_tasks':sum(t['all_five_pass'] for t in taskrows),'all_five_current_source_pass_tasks':sum(t['all_five_current_source_pass'] for t in taskrows),'current_source_large':population(cur),'cohorts':dict(collections.Counter(c['provenance_cohort'] for c in large if c['recorded'])), 'tiny_current_source_recorded':sum(c['source_current_at_capture'] for c in tiny),'tiny_current_source_pass':sum(c['source_current_at_capture'] and c['status']=='pass' for c in tiny), 'optimal_reference_strict_flags_large':sum(c['instance']!='tiny' for c in anomalies),'optimal_reference_beyond_tolerance_flags_large':sum(c['instance']!='tiny' and c['tolerance_comparison']=='better' for c in anomalies),'optimal_reference_strict_flags_tiny':sum(c['instance']=='tiny' for c in anomalies),'benchmark_process_count':len(lines),'running_cells':scope['summary'].get('official_large_running_cells',[]),'infrastructure_error_tasks':scope['summary'].get('infrastructure_error_tasks',[]),'resource_failure_instances':scope['summary'].get('resource_failure_instances',[]),'stale_tiny_tasks':scope['summary'].get('stale_official_tiny_source_tasks',[]), 'authoring_provenance':scope['authoring_provenance']}
    assert summary['large']['recorded']==scope['summary']['official_large_completed_rows']
    assert summary['large']['pass']==scope['summary']['official_large_pass']
    assert summary['large']['strict']['better']+summary['large']['strict']['equal']==scope['summary']['equal_or_better_large_instances']
    assert summary['all_five_pass_tasks']==scope['summary']['tasks_all_five_large_pass']
    # Preserved pilot cases are reported separately from the 179 newly authored tasks.
    assert len(cells)==1080 and len(taskrows)==180
    dump(dest/'summary.json',summary)
    audits=[]
    for tid in ['alfieri2020','amaldi2013']:
        for i in range(1,6):
            inst='large_'+str(i);e=recs[(tid,inst)]; rd=Path(e['results_csv']).parent
            solp=rd/'official/eval/eval_tasks'/tid/'gpt-5.3-codex'/('solution_'+inst+'.json')
            inputp=DATA/'tasks'/tid/'instance'/('large_instance_'+str(i)+'.json')
            folder=dest/'independent-audit'/tid/inst;folder.mkdir(parents=True)
            ib=inputp.read_bytes();ob=solp.read_bytes();(folder/'input.json').write_bytes(ib);(folder/'solution.json').write_bytes(ob)
            inp=json.loads(ib);sol=json.loads(ob);a={'task_id':tid,'instance':inst,'input_sha256':digest(ib),'solution_sha256':digest(ob),'source_current_at_capture':next(c['source_current_at_capture'] for c in cells if c['task_id']==tid and c['instance']==inst)}
            if tid=='alfieri2020':
                colors=sol['coloring'];assert set(colors)==set(map(str,range(inp['num_nodes'])))
                assert all(isinstance(v,int) and not isinstance(v,bool) and v>=0 for v in colors.values())
                conflicts=sum(colors[str(u)]==colors[str(v)] for u,v in inp['edges']);assert conflicts==0
                assert len(set(colors.values()))==sol['objective_value']
                a.update(nodes=inp['num_nodes'],edges=len(inp['edges']),colors=sol['objective_value'],edge_conflicts=conflicts)
            else:
                planes=sol['hyperplanes'];assignment=sol['point_assignments'];assert len(planes)==sol['objective_value'] and len(assignment)==inp['n']
                worst=-float('inf');maxnorm=0
                for h,plane in enumerate(planes):
                    w=plane['w'];norm=math.sqrt(sum(x*x for x in w));maxnorm=max(maxnorm,abs(norm-1));assert abs(norm-1)<=1e-6 and len(w)==inp['d']
                    ap=plane['assigned_points'];assert len(set(ap))==len(ap)
                    for j in ap:assert 0<=j<inp['n'] and h in assignment[j]
                for j,hs in enumerate(assignment):
                    assert hs and len(hs)==len(set(hs))
                    for h in hs:
                        assert isinstance(h,int) and 0<=h<len(planes) and j in planes[h]['assigned_points']
                        p=planes[h];norm=math.sqrt(sum(x*x for x in p['w']));dist=abs(sum(x*y for x,y in zip(inp['points'][j],p['w']))-p['w0'])/norm
                        worst=max(worst,dist-inp['epsilon']);assert dist<=inp['epsilon']+1e-7
                a.update(points=inp['n'],dimensions=inp['d'],planes=sol['objective_value'],maximum_norm_error=maxnorm,maximum_distance_excess=worst)
            a['independent_arithmetic_pass']=True;audits.append(a)
    dump(dest/'independent-audit.json',audits)
    # Preserve the named checker-dependency failure without executing its checker.
    for tid in summary['infrastructure_error_tasks']:
        for inst,e in scope['tasks'][tid].get('evaluations',{}).items():
            if e.get('state')=='checker_dependency_blocked':
                dump(dest/('blocked-'+tid+'-'+inst+'.json'),e)
    inventory=[]
    for p in sorted(dest.rglob('*')):
        if p.is_file():inventory.append({'path':str(p.relative_to(dest)),'bytes':p.stat().st_size,'sha256':digest(p.read_bytes())})
    dump(dest/'file-manifest.json',inventory)
    archive=dest.parent/(dest.name+'-evidence.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in sorted(dest.rglob('*')):
            if p.is_file():z.write(p,str(p.relative_to(dest)))
        z.write(__file__,'capture_report.py')
    print(json.dumps(summary,indent=2));print('EVIDENCE_ZIP',archive,archive.stat().st_size)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--snapshot-dir',type=Path,default=ROOT/'snapshot-20261010');args=ap.parse_args();capture(args.snapshot_dir)
