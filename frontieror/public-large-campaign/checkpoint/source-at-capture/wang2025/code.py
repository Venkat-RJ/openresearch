import argparse,json,os,time,itertools,math
os.environ['OPENBLAS_NUM_THREADS']='1'
import numpy as np

def solve(d,limit):
 a=np.array(d['data_matrix']);n=d['n'];k=min(d['k'],n);kind=d['instance_type'];weights=np.array(d['node_weights']) if kind=='COV' else np.ones(len(a));deadline=time.monotonic()+max(.01,limit-.5)
 def val(s):
  if not s:return 0.
  if kind=='INF':return float(np.sum(1-np.prod(1-a[:,s],axis=1)))
  return float(np.dot(weights,np.max(a[:,s],axis=1)))
 s=[]
 for _ in range(k):s.append(max((i for i in range(n) if i not in s),key=lambda i:val(s+[i])))
 z=val(s)
 if math.comb(n,k)<1000000:
  for c in itertools.combinations(range(n),k):
   if time.monotonic()>deadline:break
   v=val(list(c))
   if v>z:z=v;s=list(c)
 else:
  improved=True
  while improved and time.monotonic()<deadline:
   improved=False
   for p in range(k):
    base=s[:p]+s[p+1:]
    for j in range(n):
     if j in s:continue
     v=val(base+[j])
     if v>z+1e-9:s=base+[j];z=v;improved=True;break
     if time.monotonic()>deadline:break
    if improved:break
 return {'objective_value':z,'selected_elements':sorted(s)}
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=float,default=60);p.add_argument('--log_path');a=p.parse_args()
 with open(a.solution_path,'w') as f:json.dump(solve(json.load(open(a.instance_path)),a.time_limit),f)
