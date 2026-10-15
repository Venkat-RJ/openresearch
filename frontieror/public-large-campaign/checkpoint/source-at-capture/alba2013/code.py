import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS'): os.environ[k]='1'
import argparse,json,time,random
from ortools.sat.python import cp_model
from solution_logger import SolutionLogger

def main():
 p=argparse.ArgumentParser(); p.add_argument('--instance_path',required=True); p.add_argument('--solution_path',required=True); p.add_argument('--time_limit',type=int,required=True); p.add_argument('--log_path'); a=p.parse_args()
 t=time.monotonic(); end=t+max(1,a.time_limit-3); d=json.load(open(a.instance_path)); n=d['parameters']['num_requests_n']; s=d['parameters']['num_stacks_s']; l=d['parameters']['stack_height_l']; P=d['pickup_region']['distance_matrix']; D=d['delivery_region']['distance_matrix']; rng=random.Random(0)
 log=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
 best=None
 def accept(po,do):
  nonlocal best
  x=[0]+po; y=[0]+do; v=sum(P[x[i]][x[(i+1)%len(x)]]+D[y[i]][y[(i+1)%len(y)]] for i in range(len(x)))
  if best is None or v<best['objective_value']:
   best={'objective_value':v,'pickup_tour':x,'delivery_tour':y}
   if log: log.log_solution(v,best)
 def delivery(po, mode=0):
  stacks=[po[i*l:(i+1)*l] for i in range(s)]; cur=0; out=[]
  while any(stacks):
   choices=[k for k,z in enumerate(stacks) if z]; k=min(choices,key=lambda k:D[cur][stacks[k][-1]]+(rng.random()*20 if mode else 0)); cur=stacks[k].pop(); out.append(cur)
  return out
 po=[]; left=set(range(1,n+1)); cur=0
 while left:
  cur=min(left,key=lambda j:P[cur][j]); left.remove(cur); po.append(cur)
 accept(po,delivery(po))
 if n<=35:
  m=cp_model.CpModel(); ranks=[]; arcs=[]
  for region,dm,hint in [('p',P,best['pickup_tour']),('d',D,best['delivery_tour'])]:
   r=[m.NewIntVar(0,n,'r'+region+str(i)) for i in range(n+1)]; m.Add(r[0]==0); m.AddAllDifferent(r)
   z={}; circuit=[]; hs={(hint[i],hint[(i+1)%len(hint)]) for i in range(len(hint))}
   for i in range(n+1):
    for j in range(n+1):
     if i==j: continue
     z[i,j]=m.NewBoolVar(region+str(i)+'_'+str(j)); circuit.append((i,j,z[i,j])); m.AddHint(z[i,j],int((i,j) in hs))
     if j: m.Add(r[j]==r[i]+1).OnlyEnforceIf(z[i,j])
   m.AddCircuit(circuit); ranks.append(r); arcs.append(z)
  stack=[m.NewIntVar(0,s-1,'stack'+str(i)) for i in range(n+1)]
  for k in range(s):
   bs=[]
   for i in range(1,n+1):
    b=m.NewBoolVar('assigned'+str(i)+'_'+str(k)); m.Add(stack[i]==k).OnlyEnforceIf(b); m.Add(stack[i]!=k).OnlyEnforceIf(b.Not()); bs.append(b)
   m.Add(sum(bs)<=l)
  for i in range(1,n+1):
   for j in range(i+1,n+1):
    same=m.NewBoolVar('same'+str(i)+'_'+str(j)); before=m.NewBoolVar('before'+str(i)+'_'+str(j)); m.Add(stack[i]==stack[j]).OnlyEnforceIf(same); m.Add(stack[i]!=stack[j]).OnlyEnforceIf(same.Not()); m.Add(ranks[0][i]<ranks[0][j]).OnlyEnforceIf(before); m.Add(ranks[0][i]>ranks[0][j]).OnlyEnforceIf(before.Not()); m.Add(ranks[1][i]>ranks[1][j]).OnlyEnforceIf([same,before]); m.Add(ranks[1][i]<ranks[1][j]).OnlyEnforceIf([same,before.Not()])
  m.Minimize(sum(dm[i][j]*z for dm,az in zip((P,D),arcs) for (i,j),z in az.items())); solver=cp_model.CpSolver(); solver.parameters.num_search_workers=1; solver.parameters.max_time_in_seconds=max(.1,end-time.monotonic())
  class CB(cp_model.CpSolverSolutionCallback):
   def on_solution_callback(self):
    accept(sorted(range(1,n+1),key=lambda i:self.Value(ranks[0][i])),sorted(range(1,n+1),key=lambda i:self.Value(ranks[1][i])))
  solver.Solve(m,CB())
 else:
  current=po[:]; cv=best['objective_value']
  while time.monotonic()<end:
   q=current[:]; i,j=sorted(rng.sample(range(n),2)); q[i:j+1]=reversed(q[i:j+1]); dd=delivery(q); x=[0]+q; y=[0]+dd; v=sum(P[x[k]][x[(k+1)%len(x)]]+D[y[k]][y[(k+1)%len(y)]] for k in range(len(x)))
   if v<cv or rng.random()<.002: current=q; cv=v
   accept(q,dd)
 json.dump(best,open(a.solution_path,'w'))
if __name__=='__main__': main()
