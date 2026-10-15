import argparse,json,time,os
for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
 _logger_start=time.time()
 ap=argparse.ArgumentParser();ap.add_argument('--instance_path',required=True);ap.add_argument('--solution_path',required=True);ap.add_argument('--time_limit',type=int,default=60);ap.add_argument('--log_path');a=ap.parse_args();deadline=time.monotonic()+max(.1,a.time_limit-2)
 if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')
 d=json.load(open(a.instance_path));types=d['problem_types'];primary=types[0];n=d['num_facilities'];m=d['num_customers'];cost=d['facility_cost'];demand=np.array(d['customer_demands'],dtype=float);incidence=np.zeros((n,m),dtype=np.uint8)
 for j in range(m):
  for i in d['customer_to_facilities_coverage'].get(str(j),[]):incidence[i,j]=1
 total=float(demand[incidence.any(axis=0)].sum());results={};logger=SolutionLogger(a.log_path,sense='minimize' if primary=='PSCLP' else 'maximize') if a.log_path else None
 if logger:logger.min_interval=0;logger.start_time=_logger_start
 def coverage(S):return incidence[S].any(axis=0) if S else np.zeros(m,dtype=bool)
 def emit():
  if len(results)!=len(types):return
  solution={'objective_value':results[primary]['objective_value'],'primary_problem_type':primary,'results':results};json.dump(solution,open(a.solution_path,'w'))
  if logger:logger.log_solution(solution['objective_value'],solution)
 def greedy(variant):
  S=[];covered=coverage([]);spend=0
  while True:
   candidates=[i for i in range(n) if i not in S and (variant=='PSCLP' or spend+cost[i]<=d['budget_B']+1e-8)]
   if not candidates or (variant=='PSCLP' and float(demand[covered].sum())>=d['covering_demand_D']):break
   gains=incidence@(demand*(~covered));i=max(candidates,key=lambda i:gains[i]/cost[i])
   if gains[i]<=0:break
   S.append(i);spend+=cost[i];covered|=incidence[i].astype(bool)
  cd=float(demand[covered].sum());return {'objective_value':spend if variant=='PSCLP' else cd,'open_facilities':S,'covered_demand':cd}
 for variant in types:results[variant]=greedy(variant)
 emit()
 for num,variant in enumerate(types):
  model=gp.Model();model.Params.Threads=1;model.Params.Seed=0;model.Params.NumericFocus=0;model.Params.MIPGap=1e-4;model.Params.LazyConstraints=1;y=model.addVars(n,vtype=GRB.BINARY);theta=model.addVar(lb=0,ub=total);opening=gp.quicksum(cost[i]*y[i] for i in range(n))
  if variant=='PSCLP':model.addConstr(theta>=d['covering_demand_D']);model.setObjective(opening)
  else:model.addConstr(opening<=d['budget_B']);model.setObjective(theta,GRB.MAXIMIZE)
  selected=[];mask=coverage([]);spend=0
  while True:
   candidates=[i for i in range(n) if i not in selected and (variant=='PSCLP' or spend+cost[i]<=d['budget_B']+1e-8)]
   if not candidates or (variant=='PSCLP' and float(demand[mask].sum())>=d['covering_demand_D']):break
   gains=incidence@(demand*(~mask));i=max(candidates,key=lambda i:gains[i]/cost[i])
   if gains[i]<=0:break
   selected.append(i);spend+=cost[i];mask|=incidence[i].astype(bool)
  for i in range(n):y[i].Start=int(i in selected)
  theta.Start=float(demand[mask].sum())
  def callback(md,where):
   if where!=GRB.Callback.MIPSOL:return
   S=[i for i in range(n) if md.cbGetSolution(y[i])>.5];covered=coverage(S);actual=float(demand[covered].sum())
   if md.cbGetSolution(theta)>actual+1e-6:
    gains=incidence@(demand*(~covered));md.cbLazy(theta<=actual+gp.quicksum(float(gains[i])*y[i] for i in range(n) if i not in S))
   else:
    results[variant]={'objective_value':sum(cost[i] for i in S) if variant=='PSCLP' else actual,'open_facilities':S,'covered_demand':actual};emit()
  model.Params.TimeLimit=max(.1,(deadline-time.monotonic())/(len(types)-num));model.optimize(callback)
  if not model.SolCount:raise RuntimeError('No covering incumbent for '+variant)
  selected=[i for i in range(n) if y[i].X>.5];cd=float(demand[coverage(selected)].sum());results[variant]={'objective_value':sum(cost[i] for i in selected) if variant=='PSCLP' else cd,'open_facilities':selected,'covered_demand':cd};model.dispose()
 s={'objective_value':results[primary]['objective_value'],'primary_problem_type':primary,'results':results};json.dump(s,open(a.solution_path,'w'))
 if logger:logger.log_solution(s['objective_value'],s)
if __name__=='__main__':main()
