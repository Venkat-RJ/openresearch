import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json
from ortools.sat.python import cp_model
from solution_logger import SolutionLogger
p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=int,required=True);p.add_argument('--log_path');a=p.parse_args();d=json.load(open(a.instance_path));jobs=d['jobs'];m=d['parameters']['m'];T=d['time_horizon'];model=cp_model.CpModel();starts=[];intervals=[];cost=[]
for i,j in enumerate(jobs):
 pp=j['processing_time'];s=model.new_int_var(0,T-pp,'s'+str(i));e=model.new_int_var(pp,T,'e'+str(i));iv=model.new_interval_var(s,pp,e,'iv'+str(i));t=model.new_int_var(0,T,'t'+str(i));model.add_max_equality(t,[e-j['due_date'],0]);starts.append(s);intervals.append(iv);cost.append(t*j['weight'])
model.add_cumulative(intervals,[1]*len(jobs),m);model.minimize(sum(cost));solver=cp_model.CpSolver();solver.parameters.max_time_in_seconds=max(.1,a.time_limit-2);solver.parameters.num_search_workers=1;solver.parameters.random_seed=0;logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
class CB(cp_model.CpSolverSolutionCallback):
 def on_solution_callback(self):
  free=[0]*m;prev=[0]*m;rows=[]
  for i in sorted(range(len(jobs)),key=lambda i:(self.value(starts[i]),jobs[i]['id'])):
   j=jobs[i];s=self.value(starts[i]);k=next(k for k in range(m) if free[k]<=s);e=s+j['processing_time'];w=j['weight']*max(0,e-j['due_date']);rows.append({'machine_id':k,'prev_job':prev[k],'job':j['id'],'start_time':s,'completion_time':e,'weighted_tardiness':w});free[k]=e;prev[k]=j['id']
  sol={'objective_value':sum(r['weighted_tardiness'] for r in rows),'schedule':rows};json.dump(sol,open(a.solution_path,'w'))
  if logger:logger.log_solution(sol['objective_value'],sol)
solver.solve(model,CB())
