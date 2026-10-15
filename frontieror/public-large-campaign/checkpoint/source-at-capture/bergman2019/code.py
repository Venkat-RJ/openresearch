import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ[k]='1'
import argparse,json,time
from ortools.sat.python import cp_model
from solution_logger import SolutionLogger

def main():
 p=argparse.ArgumentParser(); p.add_argument('--instance_path',required=True); p.add_argument('--solution_path',required=True); p.add_argument('--time_limit',type=int,required=True); p.add_argument('--log_path'); a=p.parse_args(); end=time.monotonic()+max(1,a.time_limit-3); d=json.load(open(a.instance_path)); n=d['n']; K=d['m']; P=d['profits']; Q=d['pairwise_profits']; W=d['weights']; C=d['capacities']; log=SolutionLogger(a.log_path,sense='maximize') if a.log_path else None; best={'objective_value':0,'assignment':[]}
 if log: log.log_solution(0,best)
 groups=[[] for _ in range(K)]; remaining=C[:]; available=set(range(n))
 while available and time.monotonic()<end:
  choice=None
  for i in available:
   for k in range(K):
    if W[i]<=remaining[k]:
     gain=P[i]+sum(Q[min(i,j)][max(i,j)] for j in groups[k]); score=gain/max(1,W[i])
     if gain>0 and (choice is None or score>choice[0]): choice=(score,i,k)
  if choice is None: break
  _,i,k=choice; available.remove(i); groups[k].append(i); remaining[k]-=W[i]
 value=sum(sum(P[i] for i in g)+sum(Q[min(i,j)][max(i,j)] for ii,i in enumerate(g) for j in g[ii+1:]) for g in groups)
 if value>0:
  best={'objective_value':value,'assignment':[[i,k] for k,g in enumerate(groups) for i in g]}
  if log: log.log_solution(value,best)
 if n>150 or time.monotonic()>=end:
  json.dump(best,open(a.solution_path,'w')); return
 m=cp_model.CpModel(); x=[[m.NewBoolVar('x'+str(i)+'_'+str(k)) for k in range(K)] for i in range(n)]
 for i in range(n): m.Add(sum(x[i])<=1)
 for k in range(K): m.Add(sum(W[i]*x[i][k] for i in range(n))<=C[k])
 terms=[P[i]*x[i][k] for i in range(n) for k in range(K)]
 for i in range(n):
  for j in range(i+1,n):
   if not Q[i][j]: continue
   for k in range(K):
    z=m.NewBoolVar('pair'+str(i)+'_'+str(j)+'_'+str(k)); m.Add(z<=x[i][k]); m.Add(z<=x[j][k]); m.Add(z>=x[i][k]+x[j][k]-1); terms.append(Q[i][j]*z)
 m.Maximize(sum(terms))
 class CB(cp_model.CpSolverSolutionCallback):
  def on_solution_callback(self):
   nonlocal best
   groups=[[i for i in range(n) if self.Value(x[i][k])] for k in range(K)]; obj=sum(sum(P[i] for i in g)+sum(Q[i][j] for ii,i in enumerate(g) for j in g[ii+1:]) for g in groups)
   if obj>=best['objective_value']:
    best={'objective_value':obj,'assignment':[[i,k] for k,g in enumerate(groups) for i in g]}
    if log: log.log_solution(obj,best)
 solver=cp_model.CpSolver(); solver.parameters.num_search_workers=1; solver.parameters.max_time_in_seconds=max(.1,end-time.monotonic()); solver.Solve(m,CB()); json.dump(best,open(a.solution_path,'w'))
if __name__=='__main__': main()
