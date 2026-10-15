"""Verify public derivative hashes and saved scientific evidence, without optimization or network.
Original byte commitments are historical. This verifier checks actual public bytes first;
original identities are used only to link source records through the release manifest.
Requires all three release ZIPs extracted into the same directory and Python >=3.10.
"""
from pathlib import Path
import hashlib,importlib.util,json,statistics,subprocess,sys,tempfile,csv,xml.etree.ElementTree as ET
R=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p,name):
 spec=importlib.util.spec_from_file_location(name,p);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module
manifest=read(R/'release-manifest.json');identities={r['path']:r for r in manifest['files']}
for row in manifest['files']:
 p=R/row['path'];assert p.is_file() and not p.is_symlink(),row['path'];assert sha(p)==row['public_sha256'] and p.stat().st_size==row['public_bytes'],row['path']
def original_identity(p):return identities[str(p.relative_to(R))]['original_sha256']
small=R/'public-small-180';campaign=read(small/'results/campaign-results.json');assert len(campaign['tasks'])==180
for row in campaign['tasks']:
 p=small/'results'/row['frozen_code'].split('/results/',1)[1]
 assert original_identity(p)==row['code_sha256'],row['task_id']
assert sum(bool(t['checker_feasible']) for t in campaign['tasks'])==177
featured=[t for t in campaign['tasks'] if t['classification']=='audited_substantive_published_score_improvement'];assert len(featured)==5
for row in featured:
 folder=small/'featured-evidence'/row['task_id'];assert original_identity(folder/'candidate.py')==row['code_sha256']
 with tempfile.TemporaryDirectory() as td:
  result=Path(td)/'result.json';run=subprocess.run([sys.executable,str(folder/'checker.py'),'--instance_path',str(folder/'input.json'),'--solution_path',str(folder/'candidate.json'),'--result_path',str(result)],capture_output=True,text=True,timeout=60)
  assert run.returncode==0,(row['task_id'],run.stderr)
  assert read(result)['feasible'],row['task_id']
large=R/'public-large-campaign/checkpoint';cases={(r['task_id'],r['instance']):r for r in csv.DictReader((large/'case-comparisons.csv').open())};checked=improvements=0
for row in read(large/'independent-audit.json'):
 task,instance=row['task_id'],row['instance'];folder=large/'independent-audit'/task/instance;inp=read(folder/'input.json');sol=read(folder/'solution.json')
 if task=='alfieri2020':
  colors={int(k):v for k,v in sol['coloring'].items()};assert set(colors)==set(range(inp['num_nodes']));assert all(isinstance(c,int) and c>=0 for c in colors.values());assert all(colors[a]!=colors[b] for a,b in inp['edges']);objective=len(set(colors.values()))
 else:
  assert task=='amaldi2013';planes=sol['hyperplanes'];covered=set()
  for plane in planes:
   assert len(plane['w'])==inp['d'];assert abs(sum(x*x for x in plane['w'])-1)<=1e-6
   for i,point in enumerate(inp['points']):
    if abs(sum(a*b for a,b in zip(plane['w'],point))-plane['w0'])<=inp['epsilon']+1e-6:covered.add(i)
  assert covered==set(range(inp['n']));objective=len(planes)
 assert objective==sol['objective_value']==float(cases[task,instance]['objective']);improvements+=objective<float(cases[task,instance]['reference']);checked+=1
assert (checked,improvements)==(10,7)
# Recheck all final saved snapshots and recompute the registered comparison statistic.
F=R/'scheduling-study/final';result=read(F/'results.json');config=read(F/'plan-before-generation.json')
assert len(result['results'])==180 and result['state']=='complete'
independent=load(F/'source/audit_schedule.py','public_audit');official=load(F/'source/official_checker.py','public_official');adapter=load(F/'source/lisa_adapter.py','public_adapter');protocol=load(F/'source/final_protocol.py','public_protocol');analysis=load(F/'source/comparator_analysis.py','public_analysis')
valid=0
for row in result['results']:
 case=F/'runs'/row['slot'];inp=read(case/'instance.json');measurement=read(case/'timing/measurement.json');objectives=[]
 for sample in measurement['samples']:
  if not sample['parseable']:continue
  captured=case/'timing'/sample['snapshot'];assert sample['observed_seconds']<=300
  sol=adapter.read_solution(ET.parse(captured).getroot(),inp) if captured.suffix=='.xml' else read(captured)
  a=independent.audit(inp,sol);b=official.check_feasibility(inp,sol);assert a['feasible'] and b['feasible'] and not a['errors'];objectives.append(a['recomputed_objective'])
 assert objectives and row['objective']==min(objectives) and row['status']=='VALID_IN_TIME';valid+=1
records=protocol.analysis_records(config['plan'],result['results']);computed=analysis.analyze(records,config['bootstrap_seed'],resamples=config['bootstrap_resamples']);assert computed==result['analysis']==read(F/'analysis.json')
wins=losses=0
for record in records:
 a=statistics.mean(r['objective'] for r in record['candidate']);b=statistics.mean(r['objective'] for r in record['comparator']);wins+=a<b;losses+=a>b
assert (wins,losses)==(29,1)
E=R/'scheduling-study/earlier';earlier=read(E/'records.json');a_check=load(E/'source/independent_checker.py','earlier_independent');b_check=load(E/'source/official_checker.py','earlier_official');earlier_valid=failures=0
for row in earlier:
 if row['objective'] is None:
  assert row['status']=='SOLVER_FAILURE';failures+=1;continue
 inp=read(E/row['instance_file']);sol=read(E/row['solution_file']);a=a_check.audit(inp,sol);b=b_check.check_feasibility(inp,sol);assert a['feasible'] and b['feasible'] and a['recomputed_objective']==row['objective'];earlier_valid+=1
assert (earlier_valid,failures)==(110,1)
report={'release':manifest['release'],'status':'passed','public_file_hashes_checked':len(manifest['files']),'selected_original_source_identities_mapped':180,'featured_saved_solutions_checked':5,'large_saved_cases_checked':10,'large_improvements':7,'final_valid_scheduling_executions':valid,'earlier_valid_schedules':earlier_valid,'retained_development_failures':failures,'fresh_instance_mean_wins':wins,'fresh_instance_mean_losses':losses,'registered_analysis_recomputed':True,'new_solver_executions':0,'network_requests':0,'scope':'Public derivative hashes and offline saved-evidence checks within this research workflow. Historical original-byte commitments were checked privately before sanitization. Not external replication.'}
print(json.dumps(report,indent=2))
