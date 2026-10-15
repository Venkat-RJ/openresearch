import argparse,json,time,os
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def solve(data,deadline,logger):
 I=data['num_machines'];J=data['num_jobs'];cost=data['cost_matrix'];res=data['resource_matrix'];cap=data['capacities'];keys=[(i,j) for i in range(I) for j in range(J) if res[i][j]<=cap[i]];m=gp.Model();m.Params.Threads=1;m.Params.Seed=0;m.Params.MIPGap=1e-4;m.Params.NumericFocus=0;m.Params.OutputFlag=0;x=m.addVars(keys,vtype=GRB.BINARY,obj={(i,j):cost[i][j] for i,j in keys})
 for j in range(J):m.addConstr(gp.quicksum(x[i,j] for i in range(I) if (i,j) in x)==1)
 for i in range(I):m.addConstr(gp.quicksum(res[i][j]*x[i,j] for j in range(J) if (i,j) in x)<=cap[i])
 remaining=cap[:];seed={};order=sorted(range(J),key=lambda j:sorted(cost[i][j] for i in range(I))[0]-sorted(cost[i][j] for i in range(I))[1])
 for j in order:
  allowed=[i for i in range(I) if remaining[i]>=res[i][j]]
  if not allowed:break
  i=min(allowed,key=lambda i:cost[i][j]);seed[j]=i;remaining[i]-=res[i][j]
 if len(seed)==J:
  for (i,j),v in x.items():v.Start=int(seed[j]==i)
 def extract(get):
  assignment={str(j):max([i for i in range(I) if (i,j) in x],key=lambda i:get(x[i,j])) for j in range(J)};return {'objective_value':sum(cost[i][int(j)] for j,i in assignment.items()),'assignments':assignment}
 def cb(model,where):
  if where==GRB.Callback.MIPSOL and logger:
   sol=extract(model.cbGetSolution);logger.log_solution(sol['objective_value'],sol)
 m.Params.TimeLimit=max(.01,deadline-time.monotonic());m.optimize(cb)
 if not m.SolCount:raise RuntimeError('no feasible assignment')
 return extract(lambda v:v.X)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit',type=int,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();data=json.load(open(a.instance_path));logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
 if logger:logger.min_interval=0;logger.start_time=time.time()-(time.monotonic()-start)
 result=solve(data,start+max(.1,a.time_limit-2),logger);result['runtime']=time.monotonic()-start
 if logger:logger.log_solution(result['objective_value'],result)
 json.dump(result,open(a.solution_path,'w'))
