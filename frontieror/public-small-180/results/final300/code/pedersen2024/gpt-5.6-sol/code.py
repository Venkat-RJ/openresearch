import time
_wall_start=time.time()
import argparse,json,heapq,os
from solution_logger import SolutionLogger
p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=int,required=True);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();end=start+a.time_limit-2
with open(a.instance_path) as f:d=json.load(f)
nodes={z['id']:z for z in d['nodes']};edges=d['edges'];alpha=d['parameters']['alpha'];Q=d['parameters']['quota'];T={i for i,z in nodes.items() if z['type']=='potential_turbine'};F={i for i,z in nodes.items() if z['type']=='substation'};root=min(F)
vc={i:alpha*z['cost']+(1-alpha)*z['scenic_impact'] for i,z in nodes.items()};profit={i:z['profit'] for i,z in nodes.items()};adj={i:[] for i in nodes};ec={}
for e in edges:
 u,v=e['from'],e['to'];w=alpha*e['cost']+(1-alpha)*e['scenic_impact'];adj[u].append((v,w));adj[v].append((u,w));ec[u,v]=ec[v,u]=w
logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
if logger:logger.min_interval=0;logger.start_time=_wall_start
in_tree={root};arcs=[];selected=set();total=0
while not F<=in_tree or total<Q:
 if time.monotonic()>=end:raise TimeoutError('quota attachment construction exceeded time limit')
 dist={i:0 for i in in_tree};pred={};heap=[(0,i) for i in in_tree];heapq.heapify(heap)
 while heap:
  dv,v=heapq.heappop(heap)
  if dv!=dist[v]:continue
  for u,w in adj[v]:
   nd=dv+w+(0 if u in in_tree else vc[u])
   if nd<dist.get(u,float('inf')):dist[u]=nd;pred[u]=v;heapq.heappush(heap,(nd,u))
 missing=F-in_tree
 if missing:t=min(missing,key=lambda i:dist[i])
 else:t=min(T-in_tree,key=lambda i:dist[i]/min(profit[i],Q-total))
 path=[];v=t
 while v not in in_tree:path.append((pred[v],v));v=pred[v]
 for u,v in reversed(path):
  in_tree.add(v);arcs.append((u,v))
  if v in T:selected.add(v);total+=profit[v]
def pack(ars,sel):return {'objective_value':sum(ec[u,v] for u,v in ars)+sum(vc[i] for i in sel),'selected_edges':[{'from':u,'to':v} for u,v in ars],'selected_turbines':sorted(sel)}
best=pack(arcs,selected)
if logger:logger.log_solution(best['objective_value'],best)
if len(edges)*4+len(T)<=1900 and time.monotonic()<end:
 import gurobipy as gp
 from gurobipy import GRB
 if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):del os.environ['GRB_LICENSE_FILE']
 ars=list(ec);m=gp.Model();m.Params.Seed=0;m.Params.MIPGap=1e-4;m.Params.NumericFocus=0;m.Params.Threads=1;m.Params.OutputFlag=0;m.Params.TimeLimit=max(.1,end-time.monotonic());x=m.addVars(ars,vtype=GRB.BINARY);f=m.addVars(ars,lb=0);y=m.addVars(list(T),vtype=GRB.BINARY)
 m.addConstr(gp.quicksum(profit[i]*y[i] for i in T)>=Q)
 for i in nodes:
  if i!=root:m.addConstr(gp.quicksum(f[u,v] for u,v in ars if v==i)-gp.quicksum(f[u,v] for u,v in ars if u==i)==(y[i] if i in T else int(i in F)))
 for u,v in ars:
  m.addConstr(f[u,v]<=(len(T)+len(F))*x[u,v])
  if v in T:m.addConstr(x[u,v]<=y[v])
 m.setObjective(gp.quicksum(ec[ar]*x[ar] for ar in ars)+gp.quicksum(vc[i]*y[i] for i in T))
 for ar in ars:x[ar].Start=int(ar in arcs)
 for i in T:y[i].Start=int(i in selected)
 def incumbent(model,where):
  if where==GRB.Callback.MIPSOL and logger:
   v=model.cbGetSolution;s=pack([ar for ar in ars if v(x[ar])>.5],{i for i in T if v(y[i])>.5});logger.log_solution(s['objective_value'],s)
 m.optimize(incumbent)
 if m.SolCount:
  s=pack([ar for ar in ars if x[ar].X>.5],{i for i in T if y[i].X>.5})
  if s['objective_value']<best['objective_value']:best=s
best['runtime']=time.monotonic()-start
if logger:logger.log_solution(best['objective_value'],best)
with open(a.solution_path,'w') as f:json.dump(best,f)
