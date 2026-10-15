import argparse,json,time,os
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def solve(data,deadline,logger):
 edges={e['edge_id']:e for e in data['edges']};R=[e for e in edges if edges[e]['is_required']];K=range(data['fleet']['num_vehicles']);N=range(data['graph']['num_nodes']);depot=data['depot'];cap=data['fleet']['vehicle_capacity'];incident={v:[e for e in edges if v in edges[e]['endpoints']] for v in N}
 m=gp.Model();m.Params.Threads=1;m.Params.Seed=0;m.Params.MIPGap=1e-4;m.Params.NumericFocus=0;m.Params.OutputFlag=0;m.Params.LazyConstraints=1
 x=m.addVars(list(K),R,vtype=GRB.BINARY);y=m.addVars(list(K),list(edges),vtype=GRB.INTEGER,ub=2);p=m.addVars(list(K),list(N),vtype=GRB.INTEGER)
 for e in R:m.addConstr(gp.quicksum(x[k,e] for k in K)==1)
 for k in K:
  m.addConstr(gp.quicksum(edges[e]['demand']*x[k,e] for e in R)<=cap)
  for v in N:m.addConstr(gp.quicksum(y[k,e]+(x[k,e] if e in R else 0) for e in incident[v])==2*p[k,v])
 if len(K)>1:m.addConstr(x[0,R[0]]==1)
 m.setObjective(gp.quicksum(edges[e]['service_cost']*x[k,e] for k in K for e in R)+gp.quicksum(edges[e]['cost']*y[k,e] for k in K for e in edges))
 def components(k,get):
  adj={v:set() for v in N}
  for e in edges:
   if round(get(y[k,e]))+(round(get(x[k,e])) if e in R else 0):
    u,v=edges[e]['endpoints'];adj[u].add(v);adj[v].add(u)
  seen=set();out=[]
  for v in N:
   if v in seen or not adj[v]:continue
   S=set();todo=[v]
   while todo:
    u=todo.pop()
    if u in S:continue
    S.add(u);todo.extend(adj[u]-S)
   seen.update(S);out.append(S)
  return out
 def extract(get):
  routes=[];obj=0
  for k in K:
   S=next((S for S in components(k,get) if depot in S),{depot});serv=[e for e in R if get(x[k,e])>.5];dead=[{'edge_id':e,'times':round(get(y[k,e]))} for e in edges if get(y[k,e])>.5 and edges[e]['endpoints'][0] in S]
   obj+=sum(edges[e]['service_cost'] for e in serv)+sum(edges[z['edge_id']]['cost']*z['times'] for z in dead);routes.append({'vehicle':k,'serviced_edges':serv,'deadheaded_edges':dead})
  return {'objective_value':obj,'routes':routes}
 def cb(model,where):
  if where!=GRB.Callback.MIPSOL:return
  get=model.cbGetSolution;valid=True
  for k in K:
   for S in components(k,get):
    if depot in S:continue
    required=[e for e in R if all(v in S for v in edges[e]['endpoints']) and get(x[k,e])>.5]
    if not required:continue
    valid=False;cross=[e for e in edges if (edges[e]['endpoints'][0] in S)!=(edges[e]['endpoints'][1] in S)];lhs=gp.quicksum(y[k,e]+(x[k,e] if e in R else 0) for e in cross)
    for e in required:model.cbLazy(lhs>=2*x[k,e])
  if valid and logger:
   sol=extract(get);logger.log_solution(sol['objective_value'],sol)
 m.Params.TimeLimit=max(.01,deadline-time.monotonic());m.optimize(cb)
 if not m.SolCount:raise RuntimeError('no feasible CARP incumbent')
 return extract(lambda v:v.X)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit',type=int,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();data=json.load(open(a.instance_path));logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
 if logger:logger.min_interval=0;logger.start_time=time.time()-(time.monotonic()-start)
 result=solve(data,start+max(.1,a.time_limit-2),logger);result['runtime']=time.monotonic()-start
 if logger:logger.log_solution(result['objective_value'],result)
 json.dump(result,open(a.solution_path,'w'))
