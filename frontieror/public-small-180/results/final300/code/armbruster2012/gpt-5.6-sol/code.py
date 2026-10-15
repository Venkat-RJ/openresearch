import argparse,json,time,os,random
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def solve(data,deadline,logger):
 n=data['num_nodes'];edges=data['edges'];weights=data['edge_weights'];nw=data['node_weights'];F=data['bisection_capacity_F'];total=sum(nw)
 m=gp.Model();m.Params.Threads=1;m.Params.Seed=0;m.Params.MIPGap=1e-4;m.Params.NumericFocus=0;m.Params.OutputFlag=0
 x=m.addVars(n,vtype=GRB.BINARY);z=m.addVars(len(edges),vtype=GRB.BINARY);m.addConstr(x[0]==1)
 side=gp.quicksum(nw[i]*x[i] for i in range(n));m.addConstr(side<=F);m.addConstr(side>=total-F)
 for k,(i,j) in enumerate(edges):
  m.addConstr(z[k]>=x[i]-x[j]);m.addConstr(z[k]>=x[j]-x[i]);m.addConstr(z[k]<=x[i]+x[j]);m.addConstr(z[k]<=2-x[i]-x[j])
 m.setObjective(gp.quicksum(weights[k]*z[k] for k in range(len(edges))))
 def extract(get):
  part=[i for i in range(n) if get(x[i])>.5];ps=set(part);obj=sum(w for (i,j),w in zip(edges,weights) if (i in ps)!=(j in ps))
  return {'objective_value':obj,'partition_S':part,'partition_complement':[i for i in range(n) if i not in ps]}
 def cb(model,where):
  if where==GRB.Callback.MIPSOL and logger:
   sol=extract(model.cbGetSolution);logger.log_solution(sol['objective_value'],sol)
 m.Params.TimeLimit=max(.01,deadline-time.monotonic());m.optimize(cb)
 return extract(lambda v:v.X)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit',type=int,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();data=json.load(open(a.instance_path));logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
 if logger:logger.min_interval=0;logger.start_time=time.time()-(time.monotonic()-start)
 result=solve(data,start+max(.1,a.time_limit-2),logger);result['runtime']=time.monotonic()-start
 if logger:logger.log_solution(result['objective_value'],result)
 json.dump(result,open(a.solution_path,'w'))
