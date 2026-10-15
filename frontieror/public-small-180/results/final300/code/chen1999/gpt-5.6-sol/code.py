import argparse,json,os,time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
 _logger_start=time.time()
 ap=argparse.ArgumentParser();ap.add_argument('--instance_path',required=True);ap.add_argument('--solution_path',required=True);ap.add_argument('--time_limit',type=int,default=60);ap.add_argument('--log_path');a=ap.parse_args()
 if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')
 d=json.load(open(a.instance_path));n=d['num_jobs'];m=d['num_machines'];w=d['jobs']['weights'];p=d['jobs']['processing_times'];tardy=d['problem_type']!='weighted_completion_time';model=gp.Model();model.Params.Threads=1;model.Params.Seed=0;model.Params.MIPGap=1e-4;model.Params.NumericFocus=0;model.Params.TimeLimit=max(.1,a.time_limit-3)
 x=model.addVars(n,m,vtype=GRB.BINARY)
 if tardy:
  due=d['jobs']['due_dates'];order=sorted(range(n),key=lambda j:(due[j],j));z=model.addVars(n,vtype=GRB.BINARY)
  for j in range(n):model.addConstr(gp.quicksum(x[j,k] for k in range(m))+z[j]==1)
  for k in range(m):
   total=sum(p[j][k] for j in range(n))
   for h,j in enumerate(order):model.addConstr(gp.quicksum(p[i][k]*x[i,k] for i in order[:h+1])<=due[j]+total*(1-x[j,k]))
  model.setObjective(gp.quicksum(w[j]*z[j] for j in range(n)))
 else:
  for j in range(n):model.addConstr(gp.quicksum(x[j,k] for k in range(m))==1)
  obj=gp.quicksum(w[j]*p[j][k]*x[j,k] for j in range(n) for k in range(m))
  for k in range(m):
   for i in range(n):
    for j in range(i+1,n):
     both=model.addVar(lb=0,ub=1);model.addConstr(both>=x[i,k]+x[j,k]-1);obj+=min(w[j]*p[i][k],w[i]*p[j][k])*both
  model.setObjective(obj)
 logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
 if logger:logger.min_interval=0;logger.start_time=_logger_start
 def extract(get):
  schedules={};completion={};on=[];obj=0
  for k in range(m):
   seq=[j for j in range(n) if get(x[j,k])>.5];seq.sort(key=lambda j:(d['jobs']['due_dates'][j],j) if tardy else (p[j][k]/w[j],j));schedules[str(k)]=seq;ct=0
   for j in seq:ct+=p[j][k];completion[str(j)]=ct;on.append(j);obj+=w[j]*ct
  late=sorted(set(range(n))-set(on))
  if tardy:obj=sum(w[j] for j in late)
  return {'objective_value':obj,'schedule':schedules,'completion_times':completion,'tardy_jobs':late,'on_time_jobs':sorted(on)}
 def cb(md,where):
  if where==GRB.Callback.MIPSOL:
   s=extract(md.cbGetSolution);json.dump(s,open(a.solution_path,'w'))
   if logger:logger.log_solution(s['objective_value'],s)
 model.optimize(cb)
 if model.SolCount:
  final=extract(lambda v:v.X);json.dump(final,open(a.solution_path,'w'))
  if logger:logger.log_solution(final['objective_value'],final)
 else:raise RuntimeError('No scheduling incumbent')
if __name__=='__main__':main()
