import argparse,json,os,time,math
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
 _logger_start=time.time()
 ap=argparse.ArgumentParser();ap.add_argument('--instance_path',required=True);ap.add_argument('--solution_path',required=True);ap.add_argument('--time_limit',type=int,default=60);ap.add_argument('--log_path');a=ap.parse_args();start=time.monotonic()
 if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')
 d=json.load(open(a.instance_path));cap=d['c'];items=sorted([w for w,v in zip(d['weights'],d['demands']) for _ in range(v)],reverse=True);bins=[]
 for i,w in enumerate(items):
  candidates=[(cap-sum(items[j] for j in b),k) for k,b in enumerate(bins) if sum(items[j] for j in b)+w<=cap]
  if candidates:bins[min(candidates)[1]].append(i)
  else:bins.append([i])
 logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
 if logger:logger.min_interval=0;logger.start_time=_logger_start
 def save(b):
  result={'objective_value':len(b),'num_bins':len(b),'bin_assignments':[[items[j] for j in v] for v in b]};json.dump(result,open(a.solution_path,'w'))
  if logger:logger.log_solution(len(b),result)
 save(bins)
 if len(bins)==math.ceil(sum(items)/cap):return
 n=len(items);B=len(bins);model=gp.Model();model.Params.Seed=0;model.Params.Threads=1;model.Params.MIPGap=1e-4;model.Params.NumericFocus=0;x=model.addVars(n,B,vtype=GRB.BINARY);y=model.addVars(B,vtype=GRB.BINARY)
 for i in range(n):model.addConstr(gp.quicksum(x[i,k] for k in range(B))==1)
 for k in range(B):model.addConstr(gp.quicksum(items[i]*x[i,k] for i in range(n))<=cap*y[k])
 for k in range(1,B):model.addConstr(y[k-1]>=y[k])
 model.addConstr(x[0,0]==1);model.setObjective(y.sum())
 for k,b in enumerate(bins):
  y[k].Start=1
  for i in range(n):x[i,k].Start=int(i in b)
 def cb(md,where):
  if where==GRB.Callback.MIPSOL:save([[i for i in range(n) if md.cbGetSolution(x[i,k])>.5] for k in range(B) if md.cbGetSolution(y[k])>.5])
 model.Params.TimeLimit=max(.1,a.time_limit-(time.monotonic()-start)-2);model.optimize(cb)
if __name__=='__main__':main()
