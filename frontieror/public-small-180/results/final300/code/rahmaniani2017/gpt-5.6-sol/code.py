import time
_wall_start=time.time()
import argparse,json,os,itertools,random,math
os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import numpy as np
from solution_logger import SolutionLogger
p=argparse.ArgumentParser()
for k in ('instance_path','solution_path'):p.add_argument('--'+k,required=True)
p.add_argument('--time_limit',type=int,required=True);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();deadline=start+a.time_limit-.3;d=json.load(open(a.instance_path));log=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
if log:log.min_interval=0;log.start_time=_wall_start
nodes=d['nodes'];n=len(nodes);P=d['parameters'];dist=np.array(d['distances']) if 'distances' in d else np.array([[P['TC']*math.hypot(q['x']-r['x'],q['y']-r['y']) for r in nodes] for q in nodes]);fixed=np.array([q['fixed_cost'] for q in nodes]);raw=d['commodities']
if abs(P['chi']-P['delta'])<1e-9:
 merged={}
 for q in raw:
  key=tuple(sorted((q['origin'],q['destination'])));merged[key]=merged.get(key,0)+q['flow']
 K=[{'origin':o,'destination':dest,'flow':f} for (o,dest),f in merged.items()]
else:K=raw
edges=[(i,) for i in range(n)]+list(itertools.combinations(range(n),2));costs=np.empty((len(K),len(edges)))
for k,q in enumerate(K):
 for e,(i,*rest) in enumerate(edges):
  j=rest[0] if rest else i;o=q['origin'];dest=q['destination'];costs[k,e]=q['flow']*min(P['chi']*dist[o,i]+P['tau']*dist[i,j]+P['delta']*dist[j,dest],P['chi']*dist[o,j]+P['tau']*dist[i,j]+P['delta']*dist[i,dest])
cache={};best=float('inf');solution=None
masks=np.array([sum(1<<i for i in e) for e in edges],dtype=np.int64)
def value(mask):
 if mask not in cache:
  ids=np.flatnonzero((masks&mask)==masks);cache[mask]=float(costs[:,ids].min(axis=1).sum()+sum(fixed[i] for i in range(n) if mask>>i&1))
 return cache[mask]
def publish(mask):
 global best,solution
 v=value(mask)
 if v>=best:return
 hubs=[i for i in range(n) if mask>>i&1];ids=np.flatnonzero((masks&mask)==masks);chosen=ids[costs[:,ids].argmin(axis=1)]
 solution={'objective_value':v,'open_hubs':hubs,'assignments':[{'commodity':k,'origin':q['origin'],'destination':q['destination'],'hub_edge':list(edges[e]),'cost':float(costs[k,e])} for k,(q,e) in enumerate(zip(K,chosen))]};best=v
 if log:log.log_solution(v,solution)
mask=min((1<<i for i in range(n)),key=value);publish(mask);rng=random.Random(0);restart=0
while time.monotonic()<deadline:
 changed=True
 while changed and time.monotonic()<deadline:
  changed=False;v=value(mask);nextmask=mask
  for i in range(n):
   mm=mask^(1<<i)
   if mm and value(mm)<v-1e-8:v=value(mm);nextmask=mm
  if nextmask==mask:
   for i in range(n):
    if not mask>>i&1:continue
    for j in range(n):
     if mask>>j&1:continue
     mm=mask^(1<<i)^(1<<j)
     if value(mm)<v-1e-8:v=value(mm);nextmask=mm
  if nextmask!=mask:mask=nextmask;changed=True;publish(mask)
 restart+=1;mask=sum(1<<i for i in rng.sample(range(n),rng.randint(1,min(8,n))));publish(mask)
solution['runtime']=time.monotonic()-start;json.dump(solution,open(a.solution_path,'w'))
