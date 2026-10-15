import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
import argparse,json,time,random,itertools
import numpy as np
from solution_logger import SolutionLogger
p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=int,required=True);p.add_argument('--log_path');a=p.parse_args();d=json.load(open(a.instance_path));n=d['num_vertices'];dep=d['depot'];D=np.zeros((n,n))
for key,v in d['travel_costs'].items():i,j=map(int,key.split(','));D[i,j]=D[j,i]=v
markets=set(range(n))-{dep};deadline=time.monotonic()+max(.1,a.time_limit-2);logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None;best=float('inf');rng=random.Random(0)
def evaluate(ids):
 purchases={};obj=0
 for k,dem in enumerate(d['product_demands']):
  avail=[i for i in d['product_markets'][str(k)] if i in ids];avail.sort(key=lambda i:d['purchase_costs'][f'{k},{i}']);buy={}
  for i in avail:
   supply=dem if d.get('supply_type')=='unlimited' else d['supplies'][f'{k},{i}']
   q=min(dem,supply);dem-=q;buy[str(i)]=q;obj+=q*d['purchase_costs'][f'{k},{i}']
  if dem>1e-8:return float('inf'),None,None
  purchases[str(k)]=buy
 r=[dep];left=set(ids)
 while left:
  j=min(left,key=lambda j:D[r[-1],j]);r.append(j);left.remove(j)
 r.append(dep)
 for _ in range(5):
  changed=False
  for i in range(1,len(r)-2):
   for j in range(i+1,len(r)-1):
    if D[r[i-1],r[j]]+D[r[i],r[j+1]]<D[r[i-1],r[i]]+D[r[j],r[j+1]]-1e-8:r[i:j+1]=reversed(r[i:j+1]);changed=True
  if not changed:break
 obj+=sum(D[r[i],r[i+1]] for i in range(len(r)-1));return float(obj),r,purchases
def save(ids):
 global best,bestids
 if len(ids)<2:return float('inf')
 obj,r,buy=evaluate(ids)
 if obj<best:
  best=obj;bestids=set(ids);s={'objective_value':obj,'visited_markets':sorted(ids),'tour':r,'tour_edges':[[r[i],r[i+1]] for i in range(len(r)-1)],'purchases':buy};json.dump(s,open(a.solution_path,'w'))
  if logger:logger.log_solution(obj,s)
 return obj
save(markets)
if n<=20:
 for mask in range(1,1<<(n-1)):
  if time.monotonic()>deadline:break
  save({v for i,v in enumerate(sorted(markets)) if mask>>i&1})
else:
 ids=markets.copy();value=best
 while time.monotonic()<deadline:
  changed=False
  for j in rng.sample(sorted(markets),len(markets)):
   trial=ids^{j};obj=save(trial)
   if obj<value:ids=trial;value=obj;changed=True
   if time.monotonic()>deadline:break
  if not changed:
   ids=bestids.copy();ids^={rng.choice(sorted(markets))};value=save(ids)
