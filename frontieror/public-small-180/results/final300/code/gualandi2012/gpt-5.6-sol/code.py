import argparse,json,time,random
from solution_logger import SolutionLogger
p=argparse.ArgumentParser()
p.add_argument('--instance_path',required=True);p.add_argument('--solution_path',required=True);p.add_argument('--time_limit',type=int,required=True);p.add_argument('--log_path')
a=p.parse_args();start=time.monotonic();end=start+max(.1,a.time_limit-3)
z=json.load(open(a.instance_path));g=z['graph'];n=g['num_vertices'];adj=[set() for _ in range(n)]
for u,v in g['edges']:adj[u].add(v);adj[v].add(u)
logger=SolutionLogger(a.log_path,sense='minimize');best=n+1;best_col=None

logger.min_interval=0;logger.start_time=time.time()-(time.monotonic()-start)
def save(col):
 global best,best_col
 value=len(set(col))
 if value>=best:return
 best=value;best_col=col.copy();classes={str(c):[v for v in range(n) if col[v]==c] for c in sorted(set(col))}
 s={'objective_value':value,'coloring':{str(v):col[v] for v in range(n)},'color_classes':classes,'runtime':time.monotonic()-start}
 json.dump(s,open(a.solution_path,'w'));logger.log_solution(value,s)

def greedy(seed):
 rng=random.Random(seed);col=[-1]*n;remaining=set(range(n));priority=[rng.random() for _ in range(n)]
 while remaining:
  v=max(remaining,key=lambda v:(len({col[u] for u in adj[v] if col[u]>=0}),len(adj[v]&remaining),priority[v]))
  used={col[u] for u in adj[v]};c=0
  while c in used:c+=1
  col[v]=c;remaining.remove(v)
 save(col)
for seed in range(50):
 if time.monotonic()>=end:break
 greedy(seed)
col=[-1]*n

def search(remaining,used):
 if time.monotonic()>=end or used>=best:return
 if not remaining:save(col);return
 v=max(remaining,key=lambda v:(len({col[u] for u in adj[v] if col[u]>=0}),len(adj[v]&remaining),len(adj[v])))
 forbidden={col[u] for u in adj[v]};rest=remaining-{v}
 for c in range(min(used+1,best-1)):
  if c in forbidden:continue
  col[v]=c;search(rest,max(used,c+1));col[v]=-1
search(set(range(n)),0)
