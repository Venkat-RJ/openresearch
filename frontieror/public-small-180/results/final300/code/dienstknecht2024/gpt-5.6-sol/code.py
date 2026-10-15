import argparse,json,os,time
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
 _logger_start=time.time()
 ap=argparse.ArgumentParser();ap.add_argument('--instance_path',required=True);ap.add_argument('--solution_path',required=True);ap.add_argument('--time_limit',type=int,default=60);ap.add_argument('--log_path');a=ap.parse_args();start=time.monotonic()
 if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')
 d=json.load(open(a.instance_path));P=d['problem_parameters']['num_periods'];containers={c['id']:c for c in d['containers']};trades={t['id']:t for t in d['trades']};avail={p:[c for c,v in containers.items() if v['availability_start']<=p<=v['availability_end']] for p in range(1,P+1)};active=[(j,p) for j,t in trades.items() for p in range(t['start_period'],t['end_period']+1)];model=gp.Model();model.Params.Threads=1;model.Params.Seed=0;model.Params.MIPGap=1e-4;model.Params.NumericFocus=0;model.Params.LazyConstraints=1;x=model.addVars([(j,p,c) for j,p in active for c in avail[p]],vtype=GRB.BINARY);change=model.addVars([(j,p) for j,p in active if p>trades[j]['start_period']],vtype=GRB.BINARY)
 for j,p in active:
  model.addConstr(gp.quicksum(x[j,p,c] for c in avail[p])==trades[j]['container_demand'])
  if (j,p) in change:
   for c in avail[p]:model.addConstr(x[j,p,c]-x.get((j,p-1,c),0)<=change[j,p])
 for p in range(1,P+1):
  for c in avail[p]:model.addConstr(gp.quicksum(x[j,p,c] for j,t in trades.items() if (j,p,c) in x)<=1)
 model.setObjective(change.sum());logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
 if logger:logger.min_interval=0;logger.start_time=_logger_start
 def extract(get):
  assigned={str(j):{} for j in trades}
  for j,p in active:assigned[str(j)][str(p)]=[c for c in avail[p] if get(x[j,p,c])>.5]
  obj=sum(set(assigned[str(j)][str(p)])!=set(assigned[str(j)][str(p-1)]) for j,p in change);return {'objective_value':obj,'assignments':assigned}
 def cb(md,where):
  if where!=GRB.Callback.MIPSOL:return
  valid=True
  for j,p in active:
   chosen={c for c in avail[p] if md.cbGetSolution(x[j,p,c])>.5};remaining=set(chosen);components=0
   while remaining:
    components+=1;stack=[remaining.pop()]
    while stack:
     c=stack.pop();neighbors=remaining.intersection(containers[c]['adjacent_containers']);remaining-=neighbors;stack.extend(neighbors)
   if components>trades[j]['max_dispersion']:md.cbLazy(gp.quicksum(x[j,p,c] for c in chosen)<=len(chosen)-1);valid=False
  if valid:
   s=extract(md.cbGetSolution);json.dump(s,open(a.solution_path,'w'))
   if logger:logger.log_solution(s['objective_value'],s)
 model.Params.TimeLimit=max(.1,a.time_limit-(time.monotonic()-start)-2);model.optimize(cb)
 if model.SolCount:
  final=extract(lambda v:v.X);json.dump(final,open(a.solution_path,'w'))
  if logger:logger.log_solution(final['objective_value'],final)
 else:raise RuntimeError('No dispersion-feasible container assignment')
if __name__=='__main__':main()
