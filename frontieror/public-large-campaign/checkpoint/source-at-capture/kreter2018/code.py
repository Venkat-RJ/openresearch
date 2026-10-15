import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json
from ortools.sat.python import cp_model
from solution_logger import SolutionLogger
p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=int,required=True);p.add_argument('--log_path');a=p.parse_args();d=json.load(open(a.instance_path));T=d['max_project_duration'];jobs=d['activities'];model=cp_model.CpModel();starts={};ends={};iv={}
for j in jobs:
 i=j['id'];starts[i]=model.new_int_var(0,T-j['duration'],'s'+str(i));ends[i]=model.new_int_var(j['duration'],T,'e'+str(i));iv[i]=model.new_interval_var(starts[i],j['duration'],ends[i],'iv'+str(i))
for i,j in d['precedence_relations']:model.add(starts[j]>=ends[i])
model.add(starts[0]==0);caps=[]
for k,c in enumerate(d['resource_costs']):
 R=model.new_int_var(0,sum(j['resource_requirements'][k] for j in jobs),'R'+str(k));model.add_cumulative([iv[j['id']] for j in jobs if j['duration']>0],[j['resource_requirements'][k] for j in jobs if j['duration']>0],R);caps.append(R)
model.minimize(sum(c*r for c,r in zip(d['resource_costs'],caps)));solver=cp_model.CpSolver();solver.parameters.num_search_workers=1;solver.parameters.random_seed=0;solver.parameters.max_time_in_seconds=max(.1,a.time_limit-2);logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
class CB(cp_model.CpSolverSolutionCallback):
 def on_solution_callback(self):
  s={'objective_value':self.objective_value,'start_times':{str(i):self.value(v) for i,v in starts.items()},'resource_amounts':{str(k):self.value(v) for k,v in enumerate(caps)}};json.dump(s,open(a.solution_path,'w'))
  if logger:logger.log_solution(s['objective_value'],s)
solver.solve(model,CB())
