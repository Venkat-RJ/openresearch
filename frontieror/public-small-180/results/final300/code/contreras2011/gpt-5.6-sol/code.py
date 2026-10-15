import argparse,json,os,time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
 _logger_start=time.time()
 ap=argparse.ArgumentParser();ap.add_argument('--instance_path',required=True);ap.add_argument('--solution_path',required=True);ap.add_argument('--time_limit',type=int,default=60);ap.add_argument('--log_path');a=ap.parse_args();start=time.monotonic()
 if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')
 d=json.load(open(a.instance_path));n=d['n'];W=d['flow_matrix'];dist=d['distance_matrix'];O=d['outgoing_flow'];incoming=[sum(W[i][j] for i in range(n)) for j in range(n)];par=d['cost_parameters'];chi=par['collection_cost_chi'];alpha=par['transfer_cost_alpha'];delta=par['distribution_cost_delta'];model=gp.Model();model.Params.Seed=0;model.Params.Threads=1;model.Params.MIPGap=1e-4;model.Params.NumericFocus=0
 z=model.addVars(n,n,vtype=GRB.BINARY);f=model.addVars(n,n,n,lb=0)
 for i in range(n):model.addConstr(gp.quicksum(z[i,k] for k in range(n))==1)
 for k in range(n):
  model.addConstr(gp.quicksum(O[i]*z[i,k] for i in range(n))<=d['capacities'][k]*z[k,k])
  for i in range(n):model.addConstr(z[i,k]<=z[k,k]);model.addConstr(gp.quicksum(f[i,k,h] for h in range(n))==O[i]*z[i,k])
 for i in range(n):
  for h in range(n):model.addConstr(gp.quicksum(f[i,k,h] for k in range(n))==gp.quicksum(W[i][j]*z[j,h] for j in range(n)))
 model.addConstr(gp.quicksum(d['capacities'][k]*z[k,k] for k in range(n))>=d['total_flow_D'])
 model.setObjective(gp.quicksum(d['setup_costs'][k]*z[k,k] for k in range(n))+gp.quicksum((chi*O[i]*dist[i][k]+delta*incoming[i]*dist[k][i])*z[i,k] for i in range(n) for k in range(n))+gp.quicksum(alpha*dist[k][h]*f[i,k,h] for i in range(n) for k in range(n) for h in range(n)))
 logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
 if logger:logger.min_interval=0;logger.start_time=_logger_start
 def extract(get):
  assignment=[max(range(n),key=lambda k:get(z[i,k])) for i in range(n)];hubs=[i for i in range(n) if assignment[i]==i];obj=sum(d['setup_costs'][k] for k in hubs)+sum(W[i][j]*(chi*dist[i][assignment[i]]+alpha*dist[assignment[i]][assignment[j]]+delta*dist[assignment[j]][j]) for i in range(n) for j in range(n));return {'objective_value':obj,'hubs':hubs,'assignment':assignment}
 def cb(md,where):
  if where==GRB.Callback.MIPSOL:
   s=extract(md.cbGetSolution);json.dump(s,open(a.solution_path,'w'))
   if logger:logger.log_solution(s['objective_value'],s)
 model.Params.TimeLimit=max(.1,a.time_limit-(time.monotonic()-start)-2);model.optimize(cb)
 if model.SolCount:
  final=extract(lambda v:v.X);json.dump(final,open(a.solution_path,'w'))
  if logger:logger.log_solution(final['objective_value'],final)
 else:raise RuntimeError('No capacitated hub assignment incumbent')
if __name__=='__main__':main()
