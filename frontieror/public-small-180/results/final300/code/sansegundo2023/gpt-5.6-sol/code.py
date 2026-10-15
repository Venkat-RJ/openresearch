import time
_wall_start=time.time()
import argparse,json,random,sys
from solution_logger import SolutionLogger
sys.setrecursionlimit(10000)
p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=int,required=True);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();end=start+a.time_limit-1
with open(a.instance_path) as f:d=json.load(f)
n=d['num_vertices'];raw={int(k):v for k,v in d['adjacency_list'].items()};order=sorted(range(n),key=lambda i:len(raw.get(i,[])));idx={v:i for i,v in enumerate(order)};adj=[0]*n
for v,neis in raw.items():
 for u in neis:
  if v!=u and u in idx:adj[idx[v]]|=1<<idx[u]
logger=SolutionLogger(a.log_path,sense='maximize') if a.log_path else None;best=[];rng=random.Random(0)
if logger:logger.min_interval=0;logger.start_time=_wall_start
def publish(q):
 global best
 if len(q)>len(best):
  best=q[:]
  if logger:logger.log_solution(len(best),{'objective_value':len(best),'clique_vertices':[order[v] for v in best]})
for it in range(min(n,100)):
 P=(1<<n)-1;q=[]
 while P:
  vs=[];bits=P
  while bits:
   b=bits&-bits;bits-=b;v=b.bit_length()-1;vs.append(v)
  v=max(vs,key=lambda v:((adj[v]&P).bit_count(),v if it==0 else rng.random()));q.append(v);P&=adj[v]
 publish(q)
 if time.monotonic()>=end:break
calls=0
def colors(P):
 vs=[];bounds=[];col=0
 while P:
  col+=1;Q=P
  while Q:
   b=Q&-Q;v=b.bit_length()-1;vs.append(v);bounds.append(col);P^=b;Q&=~b;Q&=~adj[v]
 return vs,bounds
def expand(P,q):
 global calls
 calls+=1
 if calls%256==0 and time.monotonic()>=end:raise TimeoutError
 vs,bs=colors(P)
 for k in range(len(vs)-1,-1,-1):
  if len(q)+bs[k]<=len(best):return
  v=vs[k];PP=P&adj[v]
  if PP:expand(PP,q+[v])
  else:publish(q+[v])
  P&=~(1<<v)
try:
 if time.monotonic()<end:expand((1<<n)-1,[])
except TimeoutError:pass
sol={'objective_value':len(best),'clique_vertices':[order[v] for v in best],'runtime':time.monotonic()-start}
if logger:logger.log_solution(sol['objective_value'],sol)
with open(a.solution_path,'w') as f:json.dump(sol,f)
