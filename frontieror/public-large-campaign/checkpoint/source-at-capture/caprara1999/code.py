import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'): os.environ[k]='1'
import argparse,json,time
import numpy as np
from ortools.sat.python import cp_model

def main():
 p=argparse.ArgumentParser()
 for k in ('instance_path','solution_path','time_limit','log_path'): p.add_argument('--'+k,required=k!='log_path')
 a=p.parse_args(); deadline=time.monotonic()+max(0,float(a.time_limit)-2)
 d=json.load(open(a.instance_path)); n=d['n']; w=np.array(d['weights']); P=np.array(d['profit_matrix'],dtype=np.int64); cap=d['capacity']; diag=P.diagonal(); rng=np.random.default_rng(0)
 best=np.zeros(n,dtype=np.int64); score=0
 for run in range(40):
  if time.monotonic()>deadline: break
  x=np.zeros(n,dtype=np.int64); rem=cap; gain=diag.astype(float).copy(); noise=np.ones(n) if run==0 else rng.uniform(.7,1.3,n)
  while True:
   eligible=(x==0)&(w<=rem)
   if not eligible.any(): break
   ratio=np.where(eligible,gain*noise/w,-np.inf); i=int(ratio.argmax()); x[i]=1; rem-=w[i]; gain+=2*P[:,i]; gain[i]=-np.inf
  for _ in range(50):
   if time.monotonic()>deadline: break
   px=P@x; add=2*px+diag; remove=2*px-diag
   ins=np.flatnonzero(x==0); outs=np.flatnonzero(x)
   if not len(ins) or not len(outs): break
   delta=add[ins,None]-remove[outs][None,:]-2*P[np.ix_(ins,outs)]
   delta=np.where(w[ins,None]<=rem+w[outs][None,:],delta,-np.inf)
   ii,jj=np.unravel_index(np.argmax(delta),delta.shape)
   if delta[ii,jj]<=0: break
   i,j=ins[ii],outs[jj]; x[i]=1;x[j]=0;rem+=int(w[j]-w[i])
  val=int(x@P@x)
  if val>score: best=x.copy();score=val
 if n<=150 and time.monotonic()<deadline:
  m=cp_model.CpModel(); x=[m.NewBoolVar('x'+str(i)) for i in range(n)]; m.Add(sum(int(w[i])*x[i] for i in range(n))<=cap); obj=[int(diag[i])*x[i] for i in range(n)]
  for i in range(n):
   m.AddHint(x[i],int(best[i]))
   for j in range(i):
    if P[i,j]:
     y=m.NewBoolVar('y%d_%d'%(i,j));m.Add(y<=x[i]);m.Add(y<=x[j]);m.Add(y>=x[i]+x[j]-1);obj.append(int(2*P[i,j])*y)
  m.Maximize(sum(obj)); s=cp_model.CpSolver();s.parameters.num_search_workers=1;s.parameters.max_time_in_seconds=max(.01,deadline-time.monotonic());status=s.Solve(m)
  if status in (cp_model.FEASIBLE,cp_model.OPTIMAL):
   cand=np.array([s.Value(v) for v in x]);val=int(cand@P@cand)
   if val>score:best=cand;score=val
 json.dump({'objective_value':score,'selected_items':best.tolist()},open(a.solution_path,'w'))
if __name__=='__main__':main()
