import time
_wall_start=time.time()
import argparse,json,random,os
from solution_logger import SolutionLogger
p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=int,required=True);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();end=start+a.time_limit-2
with open(a.instance_path) as f:d=json.load(f)
n=d['num_customers'];m=d['num_vehicles'];C=d['vehicle_capacity'];G=d['uncertainty_set']['Gamma'];D=d['distance_matrix'];c={z['id']:z for z in d['customers']};rng=random.Random(0);logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
if logger:logger.min_interval=0;logger.start_time=_wall_start
def load(r):return sum(c[i]['nominal_demand'] for i in r)+sum(sorted((c[i]['demand_deviation'] for i in r),reverse=True)[:G])
def cost(r):return sum(D[i][j] for i,j in zip([0]+r,r+[0]))
def pack(rs):return {'objective_value':sum(cost(r) for r in rs),'routes':[{'customers':r,'cost':cost(r),'worst_case_demand':load(r)} for r in rs]}
best=None;attempt=0
while time.monotonic()<end and attempt<30:
 rs=[[i] for i in range(1,n+1)];attempt+=1
 while len(rs)>m and time.monotonic()<end:
  moves=[]
  for j in range(len(rs)):
   for k in range(j):
    if load(rs[j]+rs[k])>C:continue
    opts=[r+s for r in (rs[j],rs[j][::-1]) for s in (rs[k],rs[k][::-1])];rr=min(opts,key=cost);saving=cost(rs[j])+cost(rs[k])-cost(rr)
    moves.append((saving*(1 if attempt==1 else rng.uniform(.8,1.2)),j,k,rr))
  if not moves:break
  _,j,k,rr=max(moves);rs[k]=rr;rs.pop(j)
 if len(rs)!=m:continue
 # Exact route-count preserved in all neighborhood moves.
 changed=True
 while changed and time.monotonic()<end:
  changed=False
  for j,r in enumerate(rs):
   for k in range(len(r)):
    for h,s in enumerate(rs):
     if j==h or len(r)<=1:continue
     short=r[:k]+r[k+1:];i=r[k]
     if load(s+[i])>C:continue
     for pos in range(len(s)+1):
      ss=s[:pos]+[i]+s[pos:]
      if cost(short)+cost(ss)<cost(r)+cost(s):rs[j]=short;rs[h]=ss;changed=True;break
     if changed:break
    if changed:break
   if changed or time.monotonic()>=end:break
  if not changed:
   for j,r in enumerate(rs):
    for k in range(len(r)):
     for h in range(k+2,len(r)+1):
      rr=r[:k]+r[k:h][::-1]+r[h:]
      if cost(rr)<cost(r):rs[j]=rr;changed=True;break
     if changed:break
    if changed:break
 s=pack(rs)
 if best is None or s['objective_value']<best['objective_value']:
  best=s
  if logger:logger.log_solution(s['objective_value'],s)
 if n<=12:break
if best is None:raise RuntimeError('Robust savings could not reach exact fleet size')
if n<=12 and time.monotonic()<end:
 import gurobipy as gp
 from gurobipy import GRB
 import itertools
 if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):del os.environ['GRB_LICENSE_FILE']
 pool=[]
 for k in range(1,n+1):
  for subset in itertools.combinations(range(1,n+1),k):
   if load(subset)>C:continue
   r=min((list(p) for p in itertools.permutations(subset)),key=cost);pool.append(r)
   if len(pool)>1800:break
  if len(pool)>1800 or time.monotonic()>=end:break
 if len(pool)<=1800:
  z=gp.Model();z.Params.Seed=0;z.Params.MIPGap=1e-4;z.Params.NumericFocus=0;z.Params.Threads=1;z.Params.OutputFlag=0;z.Params.TimeLimit=max(.1,end-time.monotonic());x=z.addVars(len(pool),vtype=GRB.BINARY)
  for i in range(1,n+1):z.addConstr(gp.quicksum(x[k] for k,r in enumerate(pool) if i in r)==1)
  z.addConstr(x.sum()==m);z.setObjective(gp.quicksum(cost(r)*x[k] for k,r in enumerate(pool)))
  def incumbent(model,where):
   if where==GRB.Callback.MIPSOL and logger:
    s=pack([r for k,r in enumerate(pool) if model.cbGetSolution(x[k])>.5]);logger.log_solution(s['objective_value'],s)
  z.optimize(incumbent)
  if z.SolCount:
   s=pack([r for k,r in enumerate(pool) if x[k].X>.5])
   if s['objective_value']<best['objective_value']:best=s
best['runtime']=time.monotonic()-start
if logger:logger.log_solution(best['objective_value'],best)
with open(a.solution_path,'w') as f:json.dump(best,f)
