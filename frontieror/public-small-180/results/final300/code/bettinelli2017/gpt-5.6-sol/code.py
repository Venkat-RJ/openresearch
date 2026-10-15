import argparse,json,time,os
if os.environ.get('GRB_LICENSE_FILE') and not os.path.exists(os.environ['GRB_LICENSE_FILE']):os.environ.pop('GRB_LICENSE_FILE')
import gurobipy as gp
from gurobipy import GRB
from solution_logger import SolutionLogger

def solve(data,deadline,logger):
 n=data['num_items'];p=data['items']['profits'];w=data['items']['weights'];cap=data['capacity'];edges=data['conflict_graph']['edges'];adj=[set() for _ in range(n)]
 for i,j in edges:adj[i].add(j);adj[j].add(i)
 m=gp.Model();m.Params.Threads=1;m.Params.Seed=0;m.Params.MIPGap=1e-4;m.Params.NumericFocus=0;m.Params.OutputFlag=0;x=m.addVars(n,vtype=GRB.BINARY);m.setObjective(gp.quicksum(p[i]*x[i] for i in range(n)),GRB.MAXIMIZE);m.addConstr(gp.quicksum(w[i]*x[i] for i in range(n))<=cap)
 remaining={tuple(sorted(e)) for e in edges}
 while remaining:
  i,j=min(remaining);clique={i,j}
  for v in sorted(adj[i]&adj[j],key=lambda k:-len(adj[k])):
   if all(v in adj[u] for u in clique):clique.add(v)
  m.addConstr(gp.quicksum(x[i] for i in clique)<=1)
  for a in clique:
   for b in clique:
    if a<b:remaining.discard((a,b))
 chosen=set();weight=0
 for i in sorted(range(n),key=lambda i:-p[i]/w[i]):
  if weight+w[i]<=cap and not (adj[i]&chosen):chosen.add(i);weight+=w[i]
 for i in range(n):x[i].Start=int(i in chosen)
 def extract(get):
  selected=[i for i in range(n) if get(x[i])>.5];return {'objective_value':sum(p[i] for i in selected),'selected_items':selected}
 def cb(model,where):
  if where==GRB.Callback.MIPSOL and logger:
   sol=extract(model.cbGetSolution);logger.log_solution(sol['objective_value'],sol)
 m.Params.TimeLimit=max(.01,deadline-time.monotonic());m.optimize(cb);return extract(lambda v:v.X)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit',type=int,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();data=json.load(open(a.instance_path));logger=SolutionLogger(a.log_path,sense='maximize') if a.log_path else None
 if logger:logger.min_interval=0;logger.start_time=time.time()-(time.monotonic()-start)
 result=solve(data,start+max(.1,a.time_limit-2),logger);result['runtime']=time.monotonic()-start
 if logger:logger.log_solution(result['objective_value'],result)
 json.dump(result,open(a.solution_path,'w'))
