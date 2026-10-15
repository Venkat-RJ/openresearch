import os
os.environ['OPENBLAS_NUM_THREADS']='1'
os.environ['OMP_NUM_THREADS']='1'
import argparse,json,time
import numpy as np

def search(d,t):
 deadline=time.monotonic()+max(.1,t-5);n=d['n'];k=d['m'];w=np.array(d['weights'],dtype=np.int64);v=np.array(d['values'],dtype=np.int64)
 raw=np.array(d['pairwise_values'],dtype=np.int64);q=np.triu(raw,1);q=q+q.T;cap=d['capacity'];rng=np.random.default_rng(0)
 best=np.full(n,k,dtype=int);bestvalue=0;iteration=0
 while time.monotonic()<deadline:
  a=np.full(n,k,dtype=int);loads=np.zeros(k+1,dtype=np.int64);g=np.tile(v[:,None],(1,k+1));g[:,k]=0;value=0
  if iteration:
   a=best.copy();a[rng.random(n)<rng.uniform(.1,.35)]=k
   for b in range(k):
    members=np.flatnonzero(a==b);loads[b]=w[members].sum();g[:,b]=v+q[:,members].sum(axis=1)
   value=int(sum(v[i] for i in range(n) if a[i]<k)+sum(q[i,j] for i in range(n) for j in range(i) if a[i]==a[j] and a[i]<k))
  tabu=np.zeros((n,k+1),dtype=int);stale=0
  for step in range(1500):
   if time.monotonic()>=deadline:break
   old=g[np.arange(n),a];move=g-old[:,None];valid=loads[None,:]+w[:,None]<=cap;valid[:,k]=True;valid[np.arange(n),a]=False
   valid &= (tabu<=step)|(value+move>bestvalue)
   scores=np.where(valid,move,-10**12);flat=int(scores.argmax());i,b=divmod(flat,k+1);delta=int(scores[i,b]);chosen=(i,b)
   target=g[np.arange(n)[None,:],a[:,None]]
   swap=target+target.T-old[:,None]-old[None,:]-q*((a[:,None]<k).astype(int)+(a[None,:]<k).astype(int))
   diff=a[:,None]!=a[None,:];newload=loads[a][:,None]-w[:,None]+w[None,:]
   feasible=(newload<=cap)|(a[:,None]==k)
   allowed=(tabu[np.arange(n)[:,None],a[None,:]]<=step)|(value+swap>bestvalue)
   mask=diff&feasible&feasible.T&allowed&allowed.T
   ss=np.where(mask,swap,-10**12);f=int(ss.argmax());u,j=divmod(f,n)
   if ss[u,j]>delta:chosen=(u,int(a[j]),j,int(a[u]));delta=int(ss[u,j])
   if delta<=-10**11:break
   changes=[chosen[:2]] if len(chosen)==2 else [chosen[:2],chosen[2:]]
   for item,dest in changes:
    src=int(a[item]);tabu[item,src]=step+int(rng.integers(5,13))
    if src<k:g[:,src]-=q[:,item];loads[src]-=w[item]
    if dest<k:g[:,dest]+=q[:,item];loads[dest]+=w[item]
    a[item]=dest
   value+=delta;stale+=1
   if value>bestvalue:bestvalue=value;best=a.copy();stale=0
   if stale>250:break
  iteration+=1
 assignment={str(i):int(best[i]) for i in range(n) if best[i]<k}
 value=sum(d['values'][int(i)] for i in assignment)+sum(int(q[i,j]) for i in range(n) for j in range(i) if best[i]==best[j] and best[i]<k)
 return {'objective_value':value,'assignment':assignment}
from ortools.sat.python import cp_model

def solve(d,t):
 if d['n']>30:return search(d,t)
 n=d['n'];k=d['m'];w=d['weights'];v=d['values'];q=d['pairwise_values'];m=cp_model.CpModel();x=[[m.new_bool_var(f'x{i}_{j}') for j in range(k)] for i in range(n)];obj=[]
 for i in range(n):m.add(sum(x[i])<=1)
 for j in range(k):
  m.add(sum(w[i]*x[i][j] for i in range(n))<=d['capacity'])
  for i in range(n):
   obj.append(v[i]*x[i][j])
   for h in range(i):
    if not q[h][i]:continue
    z=m.new_bool_var(f'z{h}_{i}_{j}');m.add(z<=x[i][j]);m.add(z<=x[h][j]);m.add(z>=x[i][j]+x[h][j]-1);obj.append(q[h][i]*z)
 m.maximize(sum(obj));s=cp_model.CpSolver();s.parameters.max_time_in_seconds=max(.1,t-1);s.parameters.num_search_workers=1;r=s.solve(m)
 if r not in (cp_model.OPTIMAL,cp_model.FEASIBLE):raise RuntimeError('No incumbent')
 assignment={str(i):j for i in range(n) for j in range(k) if s.value(x[i][j])};value=sum(v[int(i)] for i in assignment)
 for i in range(n):
  for h in range(i):
   if str(i) in assignment and str(h) in assignment and assignment[str(i)]==assignment[str(h)]:value+=q[h][i]
 return {'objective_value':value,'assignment':assignment}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=float,default=60);p.add_argument('--log_path');a=p.parse_args();start=__import__('time').time();r=solve(json.load(open(a.instance_path)),a.time_limit);json.dump(r,open(a.solution_path,'w'))
 if a.log_path:open(a.log_path,'w').write(json.dumps({'time':__import__('time').time()-start,'objective_value':r['objective_value']})+'\n')
