import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import argparse,json,time
_PROCESS_START=time.monotonic()
from ortools.sat.python import cp_model
from solution_logger import SolutionLogger

def main():
 p=argparse.ArgumentParser()
 for k in ('instance_path','solution_path','log_path'): p.add_argument('--'+k,required=True)
 p.add_argument('--time_limit',type=int,default=60); a=p.parse_args(); start=_PROCESS_START;deadline=start+max(0.,a.time_limit-5.)
 d=json.load(open(a.instance_path)); n=d['num_nodes']; edges=d['edges']; adj=[set() for _ in range(n)]
 for u,v in edges: adj[u].add(v); adj[v].add(u)
 logger=SolutionLogger(a.log_path,sense='minimize')
 best=None;best_colors=None
 def save(c):
  nonlocal best,best_colors
  obj=len(set(c))
  if best is None or obj<best:
   best=obj;best_colors=list(c); sol={'objective_value':obj,'coloring':{str(i):int(v) for i,v in enumerate(c)}}
   json.dump(sol,open(a.solution_path,'w'))
   if hasattr(logger,'log_solution'): logger.log_solution(obj,sol)
   else: logger.log(obj)
 natural=[-1]*n
 for i in range(n):
  used={natural[j] for j in adj[i]}; v=0
  while v in used:v+=1
  natural[i]=v
 save(natural)
 c=[-1]*n; left=set(range(n))
 while left:
  if time.monotonic()>=deadline:
   print('DSATUR_DEADLINE remaining_vertices',len(left),'elapsed',time.monotonic()-start,flush=True);return
  i=max(left,key=lambda x:(len({c[j] for j in adj[x] if c[j]>=0}),len(adj[x]),-x)); used={c[j] for j in adj[i]}; v=0
  while v in used:v+=1
  c[i]=v;left.remove(i)
 save(c)
 print('HEURISTIC_END elapsed',time.monotonic()-start,'objective',best,flush=True)
 if n and best>1:
  m=cp_model.CpModel(); x=[m.NewIntVar(0,best-1,str(i)) for i in range(n)]; maximum=m.NewIntVar(0,best-1,'max')
  for index,(u,v) in enumerate(edges):
   if index%500==0 and time.monotonic()>=deadline:
    print('MODEL_BUILD_DEADLINE edges_built',index,'elapsed',time.monotonic()-start,flush=True);return
   m.Add(x[u]!=x[v])
  m.AddMaxEquality(maximum,x);m.Minimize(maximum);m.Add(x[0]==0)
  for i,v in enumerate(best_colors):m.AddHint(x[i],v)
  class Callback(cp_model.CpSolverSolutionCallback):
   def on_solution_callback(self):save([self.Value(v) for v in x])
  s=cp_model.CpSolver();s.parameters.num_search_workers=1;remaining=deadline-time.monotonic()
  print('MODEL_BUILD_END elapsed',time.monotonic()-start,'solve_budget',remaining,flush=True)
  if remaining<=0:return
  s.parameters.max_time_in_seconds=remaining;s.Solve(m,Callback())
  print('SOLVE_END elapsed',time.monotonic()-start,'objective',best,flush=True)
if __name__=='__main__':main()
