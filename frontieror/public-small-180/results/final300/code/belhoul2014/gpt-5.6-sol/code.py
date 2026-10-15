import argparse,json,time,os
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def solve(data,deadline,logger):
 n=data['n'];p=data['p'];cost=data['cost_matrices'];ref=data['reference_point'];lam=data['search_direction_lambda'];m=gp.Model();m.Params.Threads=1;m.Params.Seed=0;m.Params.MIPGap=1e-4;m.Params.NumericFocus=0;m.Params.OutputFlag=0;x=m.addVars(n,n,vtype=GRB.BINARY);mu=m.addVar(lb=-GRB.INFINITY,obj=1)
 for i in range(n):m.addConstr(x.sum(i,'*')==1);m.addConstr(x.sum('*',i)==1)
 for k in range(p):m.addConstr(mu>=lam[k]*(gp.quicksum(cost[k][i][j]*x[i,j] for i in range(n) for j in range(n))-ref[k]))
 for i in range(n):
  for j in range(n):x[i,j].Start=int(i==j)
 def extract(get):
  perm=[max(range(n),key=lambda j:get(x[i,j])) for i in range(n)];obj=max(lam[k]*(sum(cost[k][i][perm[i]] for i in range(n))-ref[k]) for k in range(p));return {'objective_value':obj,'assignment':perm,'assignment_matrix':[[int(j==perm[i]) for j in range(n)] for i in range(n)]}
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
