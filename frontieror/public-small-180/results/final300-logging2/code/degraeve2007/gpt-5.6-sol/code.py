import argparse,json,os,time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
 _logger_start=time.time()
 ap=argparse.ArgumentParser();ap.add_argument('--instance_path',required=True);ap.add_argument('--solution_path',required=True);ap.add_argument('--time_limit',type=int,default=60);ap.add_argument('--log_path');a=ap.parse_args();start=time.monotonic()
 if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')
 d=json.load(open(a.instance_path));n=d['dimensions']['num_products'];T=d['dimensions']['num_periods'];p=d['parameters'];dem=p['demand'];cap=p['capacity'];st=p['setup_time'];vt=p['variable_production_time'];model=gp.Model();model.Params.Seed=0;model.Params.Threads=1;model.Params.MIPGap=1e-4;model.Params.NumericFocus=0;x=model.addVars(n,T,lb=0);y=model.addVars(n,T,vtype=GRB.BINARY);s=model.addVars(n,T,lb=0);initial=model.addVars(n,lb=0)
 for i in range(n):
  for t in range(T):model.addConstr((s[i,t-1] if t else initial[i])+x[i,t]==dem[i][t]+s[i,t]);model.addConstr(x[i,t]<=max(0,min((cap[t]-st[i][t])/vt[i][t],sum(dem[i][t:])))*y[i,t])
 for t in range(T):model.addConstr(gp.quicksum(st[i][t]*y[i,t]+vt[i][t]*x[i,t] for i in range(n))<=cap[t])
 model.setObjective(gp.quicksum(p['initial_inventory_cost'][i]*initial[i] for i in range(n))+gp.quicksum(p['setup_cost'][i][t]*y[i,t]+p['variable_production_cost'][i][t]*x[i,t]+p['holding_cost'][i][t]*s[i,t] for i in range(n) for t in range(T)));logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
 if logger:logger.min_interval=0;logger.start_time=_logger_start
 def extract(get):
  result={'production':[[float(get(x[i,t])) for t in range(T)] for i in range(n)],'setup':[[int(round(get(y[i,t]))) for t in range(T)] for i in range(n)],'inventory':[[float(get(s[i,t])) for t in range(T)] for i in range(n)],'initial_inventory':[float(get(initial[i])) for i in range(n)]};obj=sum(p['initial_inventory_cost'][i]*result['initial_inventory'][i] for i in range(n))+sum(p['setup_cost'][i][t]*result['setup'][i][t]+p['variable_production_cost'][i][t]*result['production'][i][t]+p['holding_cost'][i][t]*result['inventory'][i][t] for i in range(n) for t in range(T));return {'objective_value':obj,'variables':result}
 best_solution=None
 def emit(result):
  nonlocal best_solution
  if best_solution is None or result['objective_value']<best_solution['objective_value']:
   best_solution=result
   json.dump(result,open(a.solution_path,'w'))
   if logger:logger.log_solution(result['objective_value'],result)
 def cb(md,where):
  if where==GRB.Callback.MIPSOL:
   emit(extract(md.cbGetSolution))
 model.Params.TimeLimit=max(.1,a.time_limit-(time.monotonic()-start)-2);model.optimize(cb)
 if model.SolCount:
  emit(extract(lambda v:v.X))
  json.dump(best_solution,open(a.solution_path,'w'))
 else:raise RuntimeError('No capacitated lot sizing incumbent')
if __name__=='__main__':main()
