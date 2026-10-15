import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time,random
from ortools.sat.python import cp_model
from solution_logger import SolutionLogger
p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=int,required=True);p.add_argument('--log_path');a=p.parse_args();d=json.load(open(a.instance_path));g=d['graph'];nodes=g.get('nodes',list(range(g.get('num_vertices',g.get('num_nodes',0)))));edges=g['edges'];adj={v:set() for v in nodes}
for u,v in edges:adj[u].add(v);adj[v].add(u)
deadline=time.monotonic()+max(.1,a.time_limit-2);logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None;best=len(nodes)+1;rng=random.Random(0);bestcol={}
def save(col):
 global best,bestcol
 classes={}
 for v,c in col.items():classes.setdefault(c,[]).append(v)
 classes=[sorted(c) for c in classes.values()];obj=len(classes)
 if obj<best:
  best=obj;bestcol={v:k for k,vs in enumerate(classes) for v in vs};s={'objective_value':obj,'vertex_colors':{str(v):k for v,k in bestcol.items()},'color_classes':classes} if 'num_nodes' in g else {'objective_value':obj,'selected_sets':classes,'coloring':{str(v):k for v,k in bestcol.items()}};json.dump(s,open(a.solution_path,'w'))
  if logger:logger.log_solution(obj,s)
for it in range(30):
 col={};sat={v:set() for v in nodes};uncolored=set(nodes)
 while uncolored:
  v=max(uncolored,key=lambda v:(len(sat[v]),len(adj[v]),0 if not it else rng.random()));c=0
  while c in sat[v]:c+=1
  col[v]=c;uncolored.remove(v)
  for u in adj[v]:sat[u].add(c)
 save(col)
 if time.monotonic()>deadline:break
while best>1 and time.monotonic()<deadline:
 k=best-1;model=cp_model.CpModel();x={v:model.new_int_var(0,k-1,'c'+str(v)) for v in nodes}
 for u,v in edges:model.add(x[u]!=x[v])
 ordered=sorted(nodes,key=lambda v:-len(adj[v]));clique=[]
 for v in ordered:
  if all(v in adj[u] for u in clique):clique.append(v)
 for c,v in enumerate(clique):
  if c<k:model.add(x[v]==c)
 if len(clique)>k:break
 solver=cp_model.CpSolver();solver.parameters.num_search_workers=1;solver.parameters.max_time_in_seconds=max(.1,deadline-time.monotonic());solver.parameters.random_seed=0
 status=solver.solve(model)
 if status in (cp_model.OPTIMAL,cp_model.FEASIBLE):save({v:solver.value(x[v]) for v in nodes})
 else:break
