import argparse,json,time,os
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def solve(data,deadline,logger):
 n,K=data['n'],data['m'];profit=data['profits'];pair=data['pairwise_profits'];weight=data['weights'];cap=data['capacities'];m=gp.Model();m.Params.Threads=1;m.Params.Seed=0;m.Params.MIPGap=1e-4;m.Params.NumericFocus=0;m.Params.OutputFlag=0;x=m.addVars(n,K,vtype=GRB.BINARY)
 obj=gp.quicksum(profit[i]*x[i,k] for i in range(n) for k in range(K))
 for i in range(n):
  m.addConstr(x.sum(i,'*')<=1)
  for j in range(i+1,n):
   if not pair[i][j]:continue
   for k in range(K):
    if weight[i]+weight[j]>cap[k]:continue
    v=m.addVar(ub=1);m.addConstr(v<=x[i,k]);m.addConstr(v<=x[j,k]);m.addConstr(v>=x[i,k]+x[j,k]-1);obj+=pair[i][j]*v
 for k in range(K):m.addConstr(gp.quicksum(weight[i]*x[i,k] for i in range(n))<=cap[k])
 m.setObjective(obj,GRB.MAXIMIZE)
 for v in x.values():v.Start=0
 def extract(get):
  chosen=[[i,k] for i in range(n) for k in range(K) if get(x[i,k])>.5];assign=dict(chosen);obj=sum(profit[i] for i in assign)+sum(pair[i][j] for i in assign for j in assign if i<j and assign[i]==assign[j]);return {'objective_value':obj,'assignment':chosen}
 def cb(model,where):
  if where==GRB.Callback.MIPSOL and logger:
   sol=extract(model.cbGetSolution);logger.log_solution(sol['objective_value'],sol)
 m.Params.TimeLimit=max(.01,deadline-time.monotonic());m.optimize(cb);return extract(lambda v:v.X)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit',type=int,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();data=json.load(open(a.instance_path));logger=SolutionLogger(a.log_path,sense='maximize') if a.log_path else None
 if logger:logger.min_interval=0;logger.start_time=time.time()-(time.monotonic()-start)
 result=solve(data,start+max(.1,a.time_limit-2),logger);result['runtime']=time.monotonic()-start
 if logger:logger.log_solution(result['objective_value'],result)
 json.dump(result,open(a.solution_path,'w'))
