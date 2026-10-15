import argparse,json,time,itertools,random
from solution_logger import SolutionLogger
p=argparse.ArgumentParser();p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=int,required=True);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();end=start+max(.1,a.time_limit-3)
z=json.load(open(a.instance_path));vertices=z['vertices'];index={v:i for i,v in enumerate(vertices)};n=len(vertices);edges=sorted({tuple(sorted(e)) for e in z['edges']});budget=min(z['interdiction_budget_k'],len(edges));base=[0]*n
for u,v in edges:u=index[u];v=index[v];base[u]|=1<<v;base[v]|=1<<u
logger=SolutionLogger(a.log_path,sense='minimize');best=n+1

logger.min_interval=0;logger.start_time=time.time()-(time.monotonic()-start)
def clique(adj):
 incumbent=0
 def search(size,q):
  nonlocal incumbent
  if size+q.bit_count()<=incumbent:return
  if not q:incumbent=max(size,incumbent);return
  order=[];bounds=[];color=0;r=q
  while r:
   color+=1;available=r
   while available:
    b=available&-available;v=b.bit_length()-1;order.append(v);bounds.append(color);r^=b;available&=~(b|adj[v])
  for i in range(len(order)-1,-1,-1):
   if size+bounds[i]<=incumbent:return
   v=order[i];search(size+1,q&adj[v]);q&=~(1<<v)
 search(0,(1<<n)-1);return incumbent

def test(chosen):
 global best
 adj=base.copy()
 for e in chosen:
  u,v=edges[e];u=index[u];v=index[v];adj[u]&=~(1<<v);adj[v]&=~(1<<u)
 value=clique(adj)
 if value<best:
  best=value;s={'objective_value':value,'interdicted_edges':[list(edges[e]) for e in chosen],'runtime':time.monotonic()-start};json.dump(s,open(a.solution_path,'w'));logger.log_solution(value,s)
test([])
if n<=15:
 for chosen in itertools.combinations(range(len(edges)),budget):
  if time.monotonic()>=end:break
  test(chosen)
  if best<= (1 if budget==len(edges) else 2):break
else:
 rng=random.Random(0)
 while time.monotonic()<end:test(rng.sample(range(len(edges)),budget))
