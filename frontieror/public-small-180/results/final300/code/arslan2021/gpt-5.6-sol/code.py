import argparse,json,time,os
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def solve(data,deadline,logger):
 items=data['items'];n=len(items);pars=data['problem_parameters'];gamma=pars['uncertainty_budget'];cap=pars['knapsack_capacity']
 m=gp.Model();m.Params.Threads=1;m.Params.Seed=0;m.Params.MIPGap=1e-4;m.Params.NumericFocus=0;m.Params.OutputFlag=0
 x=m.addVars(n,vtype=GRB.BINARY);y=m.addVars(n,ub=1);r=m.addVars(n,ub=1);v=m.addVars(n);u=m.addVar()
 for i,a in enumerate(items):
  m.addConstr(u+v[i]>=a['max_degradation']*(y[i]-r[i]));m.addConstr(r[i]<=y[i]);m.addConstr(y[i]<=x[i]);x[i].Start=0;y[i].Start=0;r[i].Start=0;v[i].Start=0
 u.Start=0
 m.addConstr(gp.quicksum(a['weight']*y[i]+a['repair_capacity']*r[i] for i,a in enumerate(items))<=cap)
 m.setObjective(gp.quicksum((a['outsource_penalty']-a['nominal_profit'])*x[i]-a['outsource_penalty']*y[i]+v[i] for i,a in enumerate(items))+gamma*u)
 best={'objective_value':0.0,'x':{str(i):0 for i in range(n)}}
 if logger:logger.log_solution(0,best)
 m.Params.TimeLimit=max(.01,deadline-time.monotonic()-1);m.optimize()
 if m.SolCount:
  selected=[round(x[i].X) for i in range(n)]
  for i in range(n):x[i].LB=selected[i];x[i].UB=selected[i];x[i].VType=GRB.CONTINUOUS
  m.Params.TimeLimit=max(.01,deadline-time.monotonic());m.optimize()
  if m.SolCount:
   best={'objective_value':m.ObjVal,'x':{str(i):selected[i] for i in range(n)}}
   if logger:logger.log_solution(m.ObjVal,best)
 return best

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit',type=int,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();data=json.load(open(a.instance_path));logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
 if logger:logger.min_interval=0;logger.start_time=time.time()-(time.monotonic()-start)
 result=solve(data,start+max(.1,a.time_limit-2),logger);result['runtime']=time.monotonic()-start
 if logger:logger.log_solution(result['objective_value'],result)
 json.dump(result,open(a.solution_path,'w'))
