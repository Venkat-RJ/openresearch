import argparse,json,time,os
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def main():
 _logger_start=time.time()
 ap=argparse.ArgumentParser();ap.add_argument('--instance_path',required=True);ap.add_argument('--solution_path',required=True);ap.add_argument('--time_limit',type=int,default=60);ap.add_argument('--log_path');a=ap.parse_args();start=time.monotonic()
 if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')
 d=json.load(open(a.instance_path));n=d['dimensions']['n'];m=d['dimensions']['m'];p=d['dimensions']['p'];dem=d['demands'];b=d['setup_times'];c=d['setup_costs'];lot=d['minimum_lot_sizes'];q=d['max_setups_per_item'];M=d['max_production'];model=gp.Model();model.Params.Seed=0;model.Params.MIPGap=1e-4;model.Params.NumericFocus=0;model.Params.Threads=1;model.Params.LazyConstraints=1;prod=model.addVars(n,m,p,lb=0);stock=model.addVars(n,p,lb=0);z=model.addVars(n,m,p+1,vtype=GRB.BINARY);g=model.addVars(n,m,p,vtype=GRB.BINARY);before=model.addVars(n,m,p,lb=0);R=model.addVars(m,p,vtype=GRB.BINARY);edges=[(i,j,k,t) for i in range(n) for j in range(n) if i!=j for k in range(m) for t in range(p) if q[i][k][t] and q[j][k][t]];y={key:model.addVar(vtype=GRB.INTEGER,lb=0,ub=q[key[1]][key[2]][key[3]]) for key in edges};enter={};leave={}
 for i in range(n):
  for t in range(p):model.addConstr((stock[i,t-1] if t else 0)+gp.quicksum(prod[i,k,t] for k in range(m))==dem[i][t]+stock[i,t])
  model.addConstr(stock[i,p-1]==0)
 for k in range(m):
  for t in range(p+1):model.addConstr(gp.quicksum(z[i,k,t] for i in range(n))==1)
  for t in range(p):
   transitions=gp.quicksum(v for (i,j,kk,tt),v in y.items() if kk==k and tt==t);model.addConstr(transitions>=R[k,t]);model.addConstr(transitions<=sum(q[i][k][t] for i in range(n))*R[k,t]);model.addConstr(gp.quicksum(d['processing_time']*prod[i,k,t] for i in range(n))+gp.quicksum(b[i][j][k]*v for (i,j,kk,tt),v in y.items() if kk==k and tt==t)<=d['machine_capacities'][k][t])
   for i in range(n):
    entering=gp.quicksum(y[j,i,k,t] for j in range(n) if (j,i,k,t) in y);leaving=gp.quicksum(y[i,j,k,t] for j in range(n) if (i,j,k,t) in y);enter[i,k,t]=entering;leave[i,k,t]=leaving;model.addConstr(z[i,k,t]+entering==leaving+z[i,k,t+1]);model.addConstr(z[i,k,t]+entering>=g[i,k,t]);model.addConstr(z[i,k,t]+entering<=q[i][k][t]*g[i,k,t]);model.addConstr(prod[i,k,t]<=M[i][t]*g[i,k,t]);model.addConstr(before[i,k,t]<=prod[i,k,t]);model.addConstr(before[i,k,t]<=M[i][t]*z[i,k,t]);model.addConstr(prod[i,k,t]-before[i,k,t]>=lot[i]*(entering-z[i,k,t+1]))
 for i in range(n):
  if not lot[i]:continue
  for k in range(m):
   for t in range(p):
    for u in range(t+1,p):model.addConstr(prod[i,k,t]-before[i,k,t]+gp.quicksum(before[i,k,v] for v in range(t+1,u+1))>=lot[i]*enter[i,k,t]-M[i][t]*(gp.quicksum(R[k,v] for v in range(t+1,u))+1-R[k,u]))
    for u in range(t,p):model.addConstr(gp.quicksum(prod[i,k,v] for v in range(t,u+1))>=lot[i]*enter[i,k,t]-lot[i]*q[i][k][t]*(gp.quicksum(1-z[i,k,v] for v in range(t+1,u+1))+(z[i,k,u+1] if u<p-1 else 0)))
 model.setObjective(gp.quicksum(d['inventory_costs'][i]*stock[i,t] for i in range(n) for t in range(p))+gp.quicksum(c[i][j][k]*v for (i,j,k,t),v in y.items()));logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
 if logger:logger.min_interval=0;logger.start_time=_logger_start
 def extract(get):
  production={}
  for i in range(n):
   remaining=list(dem[i])
   for t in range(p):
    for k in range(m):
     qty=max(0.,float(get(prod[i,k,t])))
     for u in range(t,p):
      take=min(qty,remaining[u])
      if take>1e-7:production[f'x_{i}_{k}_{t}_{u}']=take;qty-=take;remaining[u]-=take
      if qty<1e-7:break
  setups={f'y_{i}_{j}_{k}_{t}':round(get(v)) for (i,j,k,t),v in y.items() if get(v)>.5};carry={f'z_{i}_{k}_{t}':1 for (i,k,t),v in z.items() if get(v)>.5};obj=sum((int(key.split('_')[4])-int(key.split('_')[3]))*d['inventory_costs'][int(key.split('_')[1])]*v for key,v in production.items())+sum(c[int(key.split('_')[1])][int(key.split('_')[2])][int(key.split('_')[3])]*v for key,v in setups.items());return {'objective_value':obj,'production':production,'setups':setups,'carryover':carry}
 def callback(md,where):
  if where!=GRB.Callback.MIPSOL:return
  valid=True
  for k in range(m):
   for t in range(p):
    active={i for i in range(n) if md.cbGetSolution(g[i,k,t])>.5};root=next(i for i in range(n) if md.cbGetSolution(z[i,k,t])>.5);reached={root};stack=[root]
    while stack:
     i=stack.pop()
     for j in range(n):
      if j not in reached and (i,j,k,t) in y and md.cbGetSolution(y[i,j,k,t])>.5:reached.add(j);stack.append(j)
    missing=active-reached
    if missing:
     cut=gp.quicksum(z[i,k,t] for i in missing)+gp.quicksum(v for (i,j,kk,tt),v in y.items() if kk==k and tt==t and i not in missing and j in missing)
     for i in missing:md.cbLazy(cut>=g[i,k,t])
     valid=False
  if valid:
   solution=extract(md.cbGetSolution);json.dump(solution,open(a.solution_path,'w'))
   if logger:logger.log_solution(solution['objective_value'],solution)
 model.Params.TimeLimit=max(.1,a.time_limit-(time.monotonic()-start)-2);model.optimize(callback)
 if model.SolCount:
  final=extract(lambda v:v.X);json.dump(final,open(a.solution_path,'w'))
  if logger:logger.log_solution(final['objective_value'],final)
 else:raise RuntimeError('No connected setup multigraph incumbent')
if __name__=='__main__':main()
