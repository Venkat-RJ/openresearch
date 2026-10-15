import argparse,json,time,os
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def solve(data,deadline,logger):
 coords={v['id']:(v['x'],v['y']) for v in data['vertices']};V=list(coords);edges=[(u,v) for i,u in enumerate(V) for v in V[i+1:]];w={(u,v):(coords[u][0]-coords[v][0])**2+(coords[u][1]-coords[v][1])**2 for u,v in edges};parent={v:v for v in V}
 def find(v):
  while parent[v]!=v:parent[v]=parent[parent[v]];v=parent[v]
  return v
 tree=[]
 for u,v in sorted(edges,key=lambda e:w[e]):
  a,b=find(u),find(v)
  if a!=b:parent[a]=b;tree.append((u,v))
 def emit(selected):
  powers={v:0 for v in V}
  for u,v in selected:powers[u]=max(powers[u],w[u,v]);powers[v]=max(powers[v],w[u,v])
  return {'objective_value':sum(powers.values()),'selected_edges':[list(e) for e in selected],'vertex_costs':{str(v):p for v,p in powers.items()}}
 best=emit(tree);UB=best['objective_value'];lb={v:min(w[e] for e in edges if v in e) for v in V};base=sum(lb.values());edges=[e for e in edges if 2*w[e]+base-lb[e[0]]-lb[e[1]]<=UB+1e-8]
 if logger:logger.log_solution(UB,best)
 if len(edges)+len(V)>1900:
  edges=[e for e in edges if e in tree or sum(1 for f in edges if e[0] in f and w[f]<w[e])<12 or sum(1 for f in edges if e[1] in f and w[f]<w[e])<12]
 m=gp.Model();m.Params.Threads=1;m.Params.Seed=0;m.Params.MIPGap=1e-4;m.Params.NumericFocus=0;m.Params.OutputFlag=0;m.Params.LazyConstraints=1;x=m.addVars(edges,vtype=GRB.BINARY);p={v:m.addVar(lb=lb[v],obj=1) for v in V}
 for u,v in edges:m.addConstr(p[u]>=w[u,v]*x[u,v]);m.addConstr(p[v]>=w[u,v]*x[u,v]);x[u,v].Start=int((u,v) in tree)
 for v in V:m.addConstr(gp.quicksum(x[e] for e in edges if v in e)>=1);p[v].Start=best['vertex_costs'][str(v)]
 m.addConstr(gp.quicksum(p.values())<=UB)
 def cb(model,where):
  nonlocal best
  if where!=GRB.Callback.MIPSOL:return
  vals=model.cbGetSolution(x);selected=[e for e in edges if vals[e]>.5];adj={v:set() for v in V}
  for u,v in selected:adj[u].add(v);adj[v].add(u)
  remain=set(V);comps=[]
  while remain:
   todo=[next(iter(remain))];comp=set()
   while todo:
    u=todo.pop()
    if u in comp:continue
    comp.add(u);todo.extend(adj[u]-comp)
   remain-=comp;comps.append(comp)
  if len(comps)>1:
   for comp in comps:model.cbLazy(gp.quicksum(x[u,v] for u,v in edges if (u in comp)!=(v in comp))>=1)
  else:
   sol=emit(selected)
   if sol['objective_value']<best['objective_value']:
    best=sol
    if logger:logger.log_solution(sol['objective_value'],sol)
 m.Params.TimeLimit=max(.01,deadline-time.monotonic());m.optimize(cb);return best

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit',type=int,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();data=json.load(open(a.instance_path));logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
 if logger:logger.min_interval=0;logger.start_time=time.time()-(time.monotonic()-start)
 result=solve(data,start+max(.1,a.time_limit-2),logger);result['runtime']=time.monotonic()-start
 if logger:logger.log_solution(result['objective_value'],result)
 json.dump(result,open(a.solution_path,'w'))
