import argparse,json,time
from ortools.sat.python import cp_model

def solve(d,t):
 g=d['graph'];n=g['num_vertices'];adj=[set() for _ in range(n)]
 for u,v in g['edges']:adj[u].add(v);adj[v].add(u)
 col={};sat=[set() for _ in range(n)]
 while len(col)<n:
  v=max((i for i in range(n) if i not in col),key=lambda i:(len(sat[i]),len(adj[i])));c=0
  while c in sat[v]:c+=1
  col[v]=c
  for u in adj[v]:sat[u].add(c)
 upper=1+max(col.values());m=cp_model.CpModel();x=[m.new_int_var(0,upper-1,f'c{i}') for i in range(n)];z=m.new_int_var(0,upper-1,'z')
 for i in range(n):m.add(x[i]<=z);m.add_hint(x[i],col[i])
 for u,v in g['edges']:m.add(x[u]!=x[v])
 m.minimize(z);s=cp_model.CpSolver();s.parameters.num_search_workers=1;s.parameters.max_time_in_seconds=max(.1,t-1);r=s.solve(m)
 if r in (cp_model.OPTIMAL,cp_model.FEASIBLE):col={i:s.value(x[i]) for i in range(n)}
 classes={}
 for i,c in col.items():classes.setdefault(str(c),[]).append(i)
 return {'objective_value':len(classes),'coloring':{str(i):c for i,c in col.items()},'color_classes':classes}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=float,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.time();r=solve(json.load(open(a.instance_path)),a.time_limit);json.dump(r,open(a.solution_path,'w'))
 if a.log_path:open(a.log_path,'w').write(json.dumps({'time':time.time()-start,'objective_value':r['objective_value']})+'\n')
