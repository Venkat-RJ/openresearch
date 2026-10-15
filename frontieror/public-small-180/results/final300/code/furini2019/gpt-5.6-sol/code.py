import argparse,json,time
from solution_logger import SolutionLogger
p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=int,required=True);p.add_argument('--log_path');a=p.parse_args()
start=time.monotonic();end=start+max(.1,a.time_limit-3);z=json.load(open(a.instance_path));n=z['num_vertices'];adj=[0]*n
for u,v in z['edges']:adj[u]|=1<<v;adj[v]|=1<<u
logger=SolutionLogger(a.log_path,sense='maximize');best=[]

logger.min_interval=0;logger.start_time=time.time()-(time.monotonic()-start)
def save(c):
 global best
 if len(c)<=len(best):return
 best=c.copy();s={'objective_value':len(c),'clique_vertices':c,'runtime':time.monotonic()-start};json.dump(s,open(a.solution_path,'w'));logger.log_solution(len(c),s)
save([0])
for root in sorted(range(n),key=lambda v:adj[v].bit_count(),reverse=True):
 c=[root];q=adj[root]
 while q:
  v=max((v for v in range(n) if q>>v&1),key=lambda v:(adj[v]&q).bit_count());c.append(v);save(c);q&=adj[v]
 save(c)

def colors(q):
 order=[];bounds=[];color=0
 while q:
  color+=1;available=q
  while available:
   b=available&-available;v=b.bit_length()-1;order.append(v);bounds.append(color);q^=b;available&=~(b|adj[v])
 return order,bounds

def expand(c,q):
 save(c)
 if time.monotonic()>=end:return
 order,bounds=colors(q)
 for i in range(len(order)-1,-1,-1):
  if len(c)+bounds[i]<=len(best):return
  v=order[i];r=q&adj[v]
  if r:expand(c+[v],r)
  else:save(c+[v])
  q&=~(1<<v)
expand([], (1<<n)-1)
