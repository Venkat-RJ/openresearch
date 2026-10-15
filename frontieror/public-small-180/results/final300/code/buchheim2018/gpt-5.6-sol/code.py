import argparse,json,time,os,heapq
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def solve(data,deadline,logger):
 arcs={a['id']:a for a in data['arcs']};N=data['num_nodes'];source=data['source_node'];target=data['target_node'];L=data['linear_costs'];Q=data['quadratic_costs'];out={v:[] for v in range(N)};inc={v:[] for v in range(N)}
 for i,a in arcs.items():out[a['from_node']].append(i);inc[a['to_node']].append(i)
 dist={source:0};parent={};heap=[(0,source)]
 while heap:
  cost,v=heapq.heappop(heap)
  if cost!=dist[v]:continue
  for aid in out[v]:
   w=arcs[aid]['to_node'];c=cost+L[aid]+Q[aid][aid]
   if c<dist.get(w,float('inf')):dist[w]=c;parent[w]=aid;heapq.heappush(heap,(c,w))
 seed=[];v=target
 while v!=source:aid=parent[v];seed.append(aid);v=arcs[aid]['from_node']
 def result(chosen):return {'objective_value':sum(L[a] for a in chosen)+sum(Q[a][b] for a in chosen for b in chosen),'solution_arcs':[arcs[a] for a in chosen]}
 best=result(seed)
 if logger:logger.log_solution(best['objective_value'],best)
 m=gp.Model();m.Params.Threads=1;m.Params.Seed=0;m.Params.MIPGap=1e-4;m.Params.NumericFocus=0;m.Params.OutputFlag=0;x=m.addVars(list(arcs),vtype=GRB.BINARY)
 for v in range(N):
  if v==source:m.addConstr(gp.quicksum(x[a] for a in out[v])==1)
  elif v==target:m.addConstr(gp.quicksum(x[a] for a in inc[v])==1)
  else:m.addConstr(gp.quicksum(x[a] for a in out[v])==gp.quicksum(x[a] for a in inc[v]))
 m.setObjective(gp.quicksum((L[a]+Q[a][a])*x[a] for a in arcs)+gp.quicksum((Q[a][b]+Q[b][a])*x[a]*x[b] for a in arcs for b in arcs if a<b and Q[a][b]+Q[b][a]))
 for a in arcs:x[a].Start=int(a in seed)
 def cb(model,where):
  nonlocal best
  if where==GRB.Callback.MIPSOL:
   sol=result([a for a in arcs if model.cbGetSolution(x[a])>.5])
   if sol['objective_value']<best['objective_value']:
    best=sol
    if logger:logger.log_solution(sol['objective_value'],sol)
 m.Params.TimeLimit=max(.01,deadline-time.monotonic());m.optimize(cb)
 if m.SolCount:
  sol=result([a for a in arcs if x[a].X>.5])
  if sol['objective_value']<best['objective_value']:best=sol
 return best

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit',type=int,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();data=json.load(open(a.instance_path));logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
 if logger:logger.min_interval=0;logger.start_time=time.time()-(time.monotonic()-start)
 result=solve(data,start+max(.1,a.time_limit-2),logger);result['runtime']=time.monotonic()-start
 if logger:logger.log_solution(result['objective_value'],result)
 json.dump(result,open(a.solution_path,'w'))
