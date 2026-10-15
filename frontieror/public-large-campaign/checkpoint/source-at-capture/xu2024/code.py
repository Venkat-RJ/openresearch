import argparse,json,os,time,itertools,math,random
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np

def solve(d,limit):
 Q=np.array(d['Q']);r=np.array(d['r']);b=d['b'];n=d['n'];k=min(n,d['k']);deadline=time.monotonic()+max(.01,limit-.5);best=None;bv=float('inf');rng=random.Random(0)
 def support(s):
  qq=Q[np.ix_(s,s)];rr=r[s];e=np.ones(len(s));A=np.column_stack((e,rr));invA=np.linalg.solve(qq,A);G=A.T@invA;x=invA[:,0]/G[0,0]
  if rr@x<b-1e-10:
   if len(s)<2 or abs(np.linalg.det(G))<1e-15:return None,float('inf')
   x=invA@np.linalg.solve(G,np.array([1.,b+1e-10]))
  return x,float(.5*x@qq@x)
 def consider(s):
  nonlocal best,bv
  x,z=support(s)
  if z<bv:best=(list(s),x);bv=z
  return z
 if k==1:
  for i in range(n):
   if r[i]>=b:consider([i])
 elif math.comb(n,k)<=1000000:
  for s in itertools.combinations(range(n),k):
   if time.monotonic()>deadline:break
   consider(list(s))
 else:
  starts=[list(np.argsort(np.diag(Q))[:k]),list(np.argsort(-r)[:k])]
  for _ in range(8):starts.append(rng.sample(range(n),k))
  for s in starts:
   z=consider(s);improved=True
   while improved and time.monotonic()<deadline:
    improved=False
    for p in range(k):
     for j in range(n):
      if j in s:continue
      ss=s.copy();ss[p]=j;zz=consider(ss)
      if zz<z-1e-10:s=ss;z=zz;improved=True;break
      if time.monotonic()>deadline:break
     if improved:break
   if time.monotonic()>deadline:break
 if best is None:raise RuntimeError('No feasible support found')
 out=np.zeros(n);out[best[0]]=best[1]
 return {'objective_value':float(.5*out@Q@out),'x':out.tolist()}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=float,default=60);p.add_argument('--log_path');a=p.parse_args()
 with open(a.solution_path,'w') as f:json.dump(solve(json.load(open(a.instance_path)),a.time_limit),f)
