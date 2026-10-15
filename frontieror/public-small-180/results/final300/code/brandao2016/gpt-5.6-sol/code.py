import argparse,json,time,os
from collections import Counter
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def solve(data,deadline,logger):
 items=data['items'];W=data['parameters']['W'];seed=[];load=[]
 for item in sorted(items,key=lambda item:-item['weight']):
  for _ in range(item['demand']):
   choices=[b for b in range(len(seed)) if load[b]+item['weight']<=W]
   if choices:b=max(choices,key=lambda b:load[b])
   else:b=len(seed);seed.append([]);load.append(0)
   seed[b].append(item['item_type']);load[b]+=item['weight']
 seed=[seed[b] for b in sorted(range(len(seed)),key=lambda b:-load[b])];best={'objective_value':len(seed),'num_bins':len(seed),'bins':seed};B=len(seed);I=len(items)
 if logger:logger.log_solution(B,best)
 if B*(I+1)>1900:return best
 m=gp.Model();m.Params.Threads=1;m.Params.Seed=0;m.Params.MIPGap=1e-4;m.Params.NumericFocus=0;m.Params.OutputFlag=0;y=m.addVars(B,vtype=GRB.BINARY,obj=1);x=m.addVars(I,B,vtype=GRB.INTEGER,ub={(i,b):min(items[i]['demand'],W//items[i]['weight']) for i in range(I) for b in range(B)})
 for i,item in enumerate(items):m.addConstr(gp.quicksum(x[i,b] for b in range(B))==item['demand'])
 loads=[gp.quicksum(items[i]['weight']*x[i,b] for i in range(I)) for b in range(B)]
 for b in range(B):
  m.addConstr(loads[b]<=W*y[b]);counts=Counter(seed[b]);y[b].Start=1
  for i,item in enumerate(items):x[i,b].Start=counts[item['item_type']]
  if b:m.addConstr(y[b]<=y[b-1]);m.addConstr(loads[b]<=loads[b-1])
 m.addConstr(y.sum()>=(sum(i['weight']*i['demand'] for i in items)+W-1)//W)
 def extract(get):
  bins=[[item['item_type'] for i,item in enumerate(items) for _ in range(round(get(x[i,b])))] for b in range(B) if get(y[b])>.5];return {'objective_value':len(bins),'num_bins':len(bins),'bins':bins}
 def cb(model,where):
  nonlocal best
  if where==GRB.Callback.MIPSOL:
   sol=extract(model.cbGetSolution)
   if sol['objective_value']<best['objective_value']:
    best=sol
    if logger:logger.log_solution(sol['objective_value'],sol)
 m.Params.TimeLimit=max(.01,deadline-time.monotonic());m.optimize(cb)
 if m.SolCount:
  sol=extract(lambda v:v.X)
  if sol['objective_value']<best['objective_value']:best=sol
 return best

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit',type=int,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();data=json.load(open(a.instance_path));logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
 if logger:logger.min_interval=0;logger.start_time=time.time()-(time.monotonic()-start)
 result=solve(data,start+max(.1,a.time_limit-2),logger);result['runtime']=time.monotonic()-start
 if logger:logger.log_solution(result['objective_value'],result)
 json.dump(result,open(a.solution_path,'w'))
