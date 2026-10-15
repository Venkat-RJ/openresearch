import argparse,json,os,time,math
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
 _logger_start=time.time()
 ap=argparse.ArgumentParser();ap.add_argument('--instance_path',required=True);ap.add_argument('--solution_path',required=True);ap.add_argument('--time_limit',type=int,default=60);ap.add_argument('--log_path');a=ap.parse_args();start=time.monotonic()
 if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')
 d=json.load(open(a.instance_path));n=d['num_nodes'];depot=d['depot'];cost={(i,j):c for i,j,c in d['arcs']};profit={(i,j):p for i,j,p in d['profitable_arcs']};A=list(cost);R=list(profit);out={i:[e for e in A if e[0]==i] for i in range(n)};inc={i:[e for e in A if e[1]==i] for i in range(n)}
 strong=[tuple(e) for e in d['strong_incompatibilities']];values={i:sum(profit[e]-cost[e] for e in R if e[0]==i) for i in range(n)};weak=[(e[0],e[1],e[2] if len(e)>2 else math.ceil(d.get('generation_parameters',{}).get('gamma',.01)*(values[e[0]]+values[e[1]]))) for e in d['weak_incompatibilities']]
 model=gp.Model();model.Params.Seed=0;model.Params.Threads=1;model.Params.MIPGap=1e-4;model.Params.NumericFocus=0
 x=model.addVars(A,vtype=GRB.INTEGER,lb=0);y=model.addVars(R,vtype=GRB.BINARY);f=model.addVars(A,lb=0);z=model.addVars(n,vtype=GRB.BINARY);u=model.addVars(len(weak),lb=0,ub=1)
 for e in R:model.addConstr(x[e]>=y[e]);model.addConstr(y[e]<=z[e[0]])
 for i in range(n):
  served=gp.quicksum(y[e] for e in R if e[0]==i);model.addConstr(z[i]<=served);model.addConstr(gp.quicksum(x[e] for e in out[i])==gp.quicksum(x[e] for e in inc[i]));balance=gp.quicksum(f[e] for e in inc[i])-gp.quicksum(f[e] for e in out[i]);model.addConstr(balance==(-gp.quicksum(y[e] for e in R if e[0]!=depot) if i==depot else served))
 for e in A:model.addConstr(f[e]<=len(R)*x[e])
 for i,j in strong:model.addConstr(z[i]+z[j]<=1)
 for h,(i,j,p) in enumerate(weak):model.addConstr(u[h]>=z[i]+z[j]-1)
 model.setObjective(gp.quicksum(profit[e]*y[e] for e in R)-gp.quicksum(cost[e]*x[e] for e in A)-gp.quicksum(p*u[h] for h,(_,_,p) in enumerate(weak)),GRB.MAXIMIZE)
 logger=SolutionLogger(a.log_path,sense='maximize') if a.log_path else None
 if logger:logger.min_interval=0;logger.start_time=_logger_start
 def extract(get):
  served=[e for e in R if get(y[e])>.5];counts={e:int(round(get(x[e]))) for e in A};active={i for i,j in served};obj=sum(profit[e] for e in served)-sum(cost[e]*v for e,v in counts.items())-sum(p for i,j,p in weak if i in active and j in active)
  return {'objective_value':obj,'served_arcs':[{'from':i,'to':j,'profit':profit[i,j],'cost':cost[i,j]} for i,j in served],'tour_arcs':[{'from':i,'to':j,'count':v,'cost':cost[i,j]} for (i,j),v in counts.items() if v]}
 def save(s):
  json.dump(s,open(a.solution_path,'w'))
  if logger:logger.log_solution(s['objective_value'],s)
 save({'objective_value':0,'served_arcs':[],'tour_arcs':[]})
 def cb(md,where):
  if where==GRB.Callback.MIPSOL:save(extract(md.cbGetSolution))
 model.Params.TimeLimit=max(.1,a.time_limit-(time.monotonic()-start)-2);model.optimize(cb)
 if model.SolCount:save(extract(lambda v:v.X))
if __name__=='__main__':main()
