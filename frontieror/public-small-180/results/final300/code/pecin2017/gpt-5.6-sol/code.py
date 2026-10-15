import time
_wall_start=time.time()
import argparse,json,math,random,os
from solution_logger import SolutionLogger
p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=int,required=True);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();deadline=start+a.time_limit-2
with open(a.instance_path) as f:d=json.load(f)
N=d['problem_parameters']['n_nodes'];n=d['problem_parameters']['n_customers'];Q=d['problem_parameters']['vehicle_capacity'];nodes=sorted(d['nodes'],key=lambda z:z['id']);sink=N-1
D=[[math.hypot(u['x']-v['x'],u['y']-v['y']) for v in nodes] for u in nodes];q=[u['demand'] for u in nodes];e=[u['time_window_open'] for u in nodes];l=[u['time_window_close'] for u in nodes];svc=[u['service_time'] for u in nodes]
logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
if logger:logger.min_interval=0;logger.start_time=_wall_start
def feasible(r):
 if sum(q[i] for i in r)>Q:return False
 t=e[0];prev=0
 for i in r+[sink]:
  t=max(e[i],t+svc[prev]+D[prev][i]);prev=i
  if t>l[i]+1e-7:return False
 return True
def cost(r):return sum(D[i][j] for i,j in zip([0]+r,r+[sink]))
def pack(rs):return {'objective_value':sum(cost(r) for r in rs),'routes':[[0]+r+[sink] for r in rs if r]}
rs=[]
for i in sorted(range(1,n+1),key=lambda i:l[i]):
 options=[]
 for j,r in enumerate(rs):
  for k in range(len(r)+1):
   rr=r[:k]+[i]+r[k:]
   if feasible(rr):options.append((cost(rr)-cost(r),j,rr))
 if options:
  _,j,rr=min(options,key=lambda z:z[0]);rs[j]=rr
 else:rs.append([i])
best=pack(rs)
if logger:logger.log_solution(best['objective_value'],best)
if n<=30 and time.monotonic()<deadline:
 import gurobipy as gp
 from gurobipy import GRB
 if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):del os.environ['GRB_LICENSE_FILE']
 arcs=[(i,j) for i in range(N-1) for j in range(1,N) if i!=j and q[i]+q[j]<=Q and e[i]+svc[i]+D[i][j]<=l[j]]
 m=gp.Model();m.Params.Seed=0;m.Params.MIPGap=1e-4;m.Params.NumericFocus=0;m.Params.Threads=1;m.Params.OutputFlag=0;m.Params.TimeLimit=max(.1,deadline-time.monotonic())
 x=m.addVars(arcs,vtype=GRB.BINARY);T=m.addVars(N,lb={i:e[i] for i in range(N)},ub={i:l[i] for i in range(N)});u=m.addVars(range(1,n+1),lb={i:q[i] for i in range(1,n+1)},ub=Q)
 for i in range(1,n+1):m.addConstr(gp.quicksum(x[h,j] for h,j in arcs if j==i)==1);m.addConstr(gp.quicksum(x[h,j] for h,j in arcs if h==i)==1)
 for i,j in arcs:
  m.addConstr(T[j]>=T[i]+svc[i]+D[i][j]-(l[i]+svc[i]+D[i][j]-e[j])*(1-x[i,j]))
  if i>0 and j<=n:m.addConstr(u[j]>=u[i]+q[j]-Q*(1-x[i,j]))
 m.setObjective(gp.quicksum(D[i][j]*x[i,j] for i,j in arcs))
 active={(i,j) for r in best['routes'] for i,j in zip(r,r[1:])}
 for ar in arcs:x[ar].Start=int(ar in active)
 def incumbent(model,where):
  if where==GRB.Callback.MIPSOL and logger:
   v=model.cbGetSolution;nxt={i:j for i,j in arcs if i>0 and v(x[i,j])>.5};routes=[]
   for h,j in arcs:
    if h==0 and j<=n and v(x[h,j])>.5:
     r=[]
     while j!=sink:r.append(j);j=nxt[j]
     routes.append(r)
   s=pack(routes);logger.log_solution(s['objective_value'],s)
 m.optimize(incumbent)
 if m.SolCount:
  nxt={i:j for i,j in arcs if i>0 and x[i,j].X>.5};rrs=[]
  for h,j in arcs:
   if h==0 and j<=n and x[h,j].X>.5:
    rr=[]
    while j!=sink:rr.append(j);j=nxt[j]
    rrs.append(rr)
  s=pack(rrs)
  if s['objective_value']<best['objective_value']:best=s;rs=rrs
else:
 changed=True
 while changed and time.monotonic()<deadline:
  changed=False
  for j in range(len(rs)):
   for k in range(len(rs[j])):
    if time.monotonic()>=deadline:break
    i=rs[j][k];short=rs[j][:k]+rs[j][k+1:];gain=cost(rs[j])-cost(short)
    move=None
    for h,r in enumerate(rs):
     if h==j:continue
     for pos in range(len(r)+1):
      rr=r[:pos]+[i]+r[pos:];delta=cost(rr)-cost(r)
      if delta<gain-1e-7 and feasible(rr):move=(h,rr,gain-delta);gain=delta
    if move:
     h,rr,_=move;rs[j]=short;rs[h]=rr;changed=True;break
   if changed:break
  rs=[r for r in rs if r]
  if changed:
   s=pack(rs)
   if s['objective_value']<best['objective_value']:best=s
   if logger:logger.log_solution(best['objective_value'],best)
best['runtime']=time.monotonic()-start
if logger:logger.log_solution(best['objective_value'],best)
with open(a.solution_path,'w') as f:json.dump(best,f)
