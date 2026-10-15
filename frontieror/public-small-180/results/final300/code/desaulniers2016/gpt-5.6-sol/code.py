import argparse,json,os,time,itertools
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
 _logger_start=time.time()
 ap=argparse.ArgumentParser();ap.add_argument('--instance_path',required=True);ap.add_argument('--solution_path',required=True);ap.add_argument('--time_limit',type=int,default=60);ap.add_argument('--log_path');a=ap.parse_args();start=time.monotonic()
 if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')
 d=json.load(open(a.instance_path));n=d['num_customers'];T=d['num_periods'];K=d['num_vehicles'];Q=d['vehicle_capacity'];dist=d['distance_matrix'];initial=d['initial_inventories'];cap=d['inventory_capacities'];holding=d['holding_costs'];dem=d['customer_demands'];production=d['supplier_production'];routes=[];costs=[]
 for mask in range(1,1<<n):
  S=[i+1 for i in range(n) if mask>>i&1];seq=min(itertools.permutations(S),key=lambda seq:sum(dist[i][j] for i,j in zip((0,)+seq,seq+(0,))));routes.append((0,)+seq+(0,));costs.append(sum(dist[i][j] for i,j in zip(routes[-1],routes[-1][1:])))
 model=gp.Model();model.Params.Seed=0;model.Params.Threads=1;model.Params.NumericFocus=0;model.Params.MIPGap=1e-4;R=len(routes);y=model.addVars(R,T,vtype=GRB.BINARY);q=model.addVars([(r,i,t) for r,seq in enumerate(routes) for i in seq[1:-1] for t in range(T)],lb=0);I={(i,t):model.addVar(lb=0,ub=cap[i]) for i in range(n+1) for t in range(T)}
 for t in range(T):
  model.addConstr(gp.quicksum(y[r,t] for r in range(R))<=K);model.addConstr(I[0,t]==(I[0,t-1] if t else initial[0])+production[t]-gp.quicksum(q[r,i,t] for r,seq in enumerate(routes) for i in seq[1:-1]))
  for r,seq in enumerate(routes):model.addConstr(gp.quicksum(q[r,i,t] for i in seq[1:-1])<=Q*y[r,t])
  for i in range(1,n+1):
   rs=[r for r,seq in enumerate(routes) if i in seq];qty=gp.quicksum(q[r,i,t] for r in rs);prev=I[i,t-1] if t else initial[i];model.addConstr(gp.quicksum(y[r,t] for r in rs)<=1);model.addConstr(I[i,t]==prev+qty-dem[i-1][t]);model.addConstr(prev+qty<=cap[i])
 model.setObjective(gp.quicksum(costs[r]*y[r,t] for r in range(R) for t in range(T))+gp.quicksum(holding[i]*I[i,t] for i in range(n+1) for t in range(T)));logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
 if logger:logger.min_interval=0;logger.start_time=_logger_start
 def extract(get):
  result={};obj=0
  for t in range(T):
   chosen=[r for r in range(R) if get(y[r,t])>.5];result[str(t+1)]={str(k):{'route':list(routes[r]),'deliveries':{str(i):float(get(q[r,i,t])) for i in routes[r][1:-1]}} for k,r in enumerate(chosen)};obj+=sum(costs[r] for r in chosen)+sum(holding[i]*get(I[i,t]) for i in range(n+1))
  return {'objective_value':obj,'routes':result,'inventories':{str(i):{str(t+1):float(get(I[i,t])) for t in range(T)} for i in range(n+1)}}
 def cb(md,where):
  if where==GRB.Callback.MIPSOL:
   s=extract(md.cbGetSolution);json.dump(s,open(a.solution_path,'w'))
   if logger:logger.log_solution(s['objective_value'],s)
 model.Params.TimeLimit=max(.1,a.time_limit-(time.monotonic()-start)-2);model.optimize(cb)
 if model.SolCount:
  final=extract(lambda v:v.X);json.dump(final,open(a.solution_path,'w'))
  if logger:logger.log_solution(final['objective_value'],final)
 else:raise RuntimeError('No inventory routing incumbent')
if __name__=='__main__':main()
