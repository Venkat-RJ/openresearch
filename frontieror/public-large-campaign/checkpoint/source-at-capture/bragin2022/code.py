import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[k]='1'
import argparse,json,time
from ortools.sat.python import cp_model
from solution_logger import SolutionLogger

def main():
 p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=int,default=60);p.add_argument('--log_path',required=True);a=p.parse_args();start=time.monotonic();d=json.load(open(a.instance_path));logger=SolutionLogger(a.log_path);I=d['num_machines'];J=d['num_jobs'];cost=d['cost_matrix'];res=d['resource_matrix'];cap=d['capacities'];m=cp_model.CpModel();x=[[m.new_bool_var('x%d_%d'%(i,j)) for j in range(J)] for i in range(I)]
 for j in range(J):m.add_exactly_one(x[i][j] for i in range(I))
 for i in range(I):m.add(sum(res[i][j]*x[i][j] for j in range(J))<=cap[i])
 m.minimize(sum(cost[i][j]*x[i][j] for i in range(I) for j in range(J)))
 best=float('inf')
 def save(assign):
  nonlocal best
  obj=sum(cost[i][j] for j,i in enumerate(assign))
  if obj>=best:return
  best=obj;sol={'objective_value':obj,'assignments':{str(j):i for j,i in enumerate(assign)}};json.dump(sol,open(a.solution_path,'w'));logger.log_solution(obj,sol)
 load=[0]*I;assign=[-1]*J
 for j in sorted(range(J),key=lambda j:sorted(cost[i][j] for i in range(I))[1]-min(cost[i][j] for i in range(I)) if I>1 else 0,reverse=True):
  options=[i for i in range(I) if load[i]+res[i][j]<=cap[i]]
  if not options:break
  i=min(options,key=lambda i:cost[i][j]);assign[j]=i;load[i]+=res[i][j]
 if min(assign)>=0:
  save(assign)
  for i in range(I):
   for j in range(J):m.add_hint(x[i][j],int(assign[j]==i))
 class Callback(cp_model.CpSolverSolutionCallback):
  def on_solution_callback(self):save([next(i for i in range(I) if self.value(x[i][j])) for j in range(J)])
 s=cp_model.CpSolver();s.parameters.num_search_workers=1;s.parameters.max_time_in_seconds=max(.1,a.time_limit*.95-(time.monotonic()-start));s.solve(m,Callback())
 if best==float('inf'):raise RuntimeError('No feasible generalized assignment found')
if __name__=='__main__':main()
