import argparse,json,time
import numpy as np

def solve(d,t):
 f=np.asarray(d['opening_costs']);a=np.asarray(d['allocation_costs']);w=1/a;n=len(f);deadline=time.monotonic()+max(.1,t-1)
 def value(s):return float(f[s].sum()+np.sum(1/w[s].sum(axis=0)))
 best=None;bv=float('inf')
 if n<=22:
  for mask in range(1,1<<n):
   if time.monotonic()>deadline:break
   s=np.array([i for i in range(n) if mask>>i&1]);v=value(s)
   if v<bv:best=s;bv=v
 else:
  for start in np.argsort(f+np.sum(a,axis=1))[:min(n,12)]:
   s={int(start)}
   while time.monotonic()<deadline:
    cur=value(list(s));move=None
    for i in range(n):
     ss=s^{i}
     if not ss:continue
     v=value(list(ss))
     if v<cur-1e-9:cur=v;move=ss
    if move is None:break
    s=move
   v=value(list(s))
   if v<bv:bv=v;best=np.array(sorted(s))
   if time.monotonic()>deadline:break
 if best is None:best=np.array([int(np.argmin(f+np.sum(a,axis=1)))]);bv=value(best)
 x=np.zeros_like(a);x[best]=w[best]/w[best].sum(axis=0);y=np.zeros(n,dtype=int);y[best]=1
 return {'objective_value':float(np.dot(f,y)+np.sum(a*x*x)),'variant':'quadratic','y':y.tolist(),'x':x.tolist()}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=float,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.time();r=solve(json.load(open(a.instance_path)),a.time_limit);json.dump(r,open(a.solution_path,'w'))
 if a.log_path:open(a.log_path,'w').write(json.dumps({'time':time.time()-start,'objective_value':r['objective_value']})+'\n')
