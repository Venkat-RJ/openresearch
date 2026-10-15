import argparse,json,time,os
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def solve(data,deadline,logger):
 n=data['n_clients'];h=data['n_servers'];cost=data['linear_costs'];pairs={}
 for i,j,k,l,q in data['quadratic_costs']:
  e=(int(i),int(j));f=(int(k),int(l));key=tuple(sorted((e,f)))
  if key not in pairs:pairs[key]=q
 m=gp.Model();m.Params.Threads=1;m.Params.Seed=0;m.Params.MIPGap=1e-4;m.Params.NumericFocus=0;m.Params.OutputFlag=0;x=m.addVars(n,h,vtype=GRB.BINARY)
 for i in range(n):m.addConstr(x.sum(i,'*')==1)
 obj=gp.quicksum(cost[i][j]*x[i,j] for i in range(n) for j in range(h))
 for (e,f),q in pairs.items():
  if e==f:obj+=q*x[e];continue
  if e[0]==f[0]:continue
  y=m.addVar(ub=1);m.addConstr(y>=x[e]+x[f]-1);m.addConstr(y<=x[e]);m.addConstr(y<=x[f]);obj+=q*y
 m.setObjective(obj)
 for i in range(n):
  chosen=min(range(h),key=lambda j:cost[i][j])
  for j in range(h):x[i,j].Start=int(j==chosen)
 def extract(get):
  assignment={i:max(range(h),key=lambda j:get(x[i,j])) for i in range(n)};obj=sum(cost[i][j] for i,j in assignment.items())+sum(q for (e,f),q in pairs.items() if assignment[e[0]]==e[1] and assignment[f[0]]==f[1]);return {'objective_value':obj,'assignment':{str(i):j for i,j in assignment.items()}}
 def cb(model,where):
  if where==GRB.Callback.MIPSOL and logger:
   sol=extract(model.cbGetSolution);logger.log_solution(sol['objective_value'],sol)
 m.Params.TimeLimit=max(.01,deadline-time.monotonic());m.optimize(cb);return extract(lambda v:v.X)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit',type=int,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();data=json.load(open(a.instance_path));logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
 if logger:logger.min_interval=0;logger.start_time=time.time()-(time.monotonic()-start)
 result=solve(data,start+max(.1,a.time_limit-2),logger);result['runtime']=time.monotonic()-start
 if logger:logger.log_solution(result['objective_value'],result)
 json.dump(result,open(a.solution_path,'w'))
