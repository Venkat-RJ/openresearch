import argparse,json,time,random,heapq
from solution_logger import SolutionLogger

def solve(data,deadline,logger):
 n=data['num_nodes']; adj=[set() for _ in range(n)]
 for u,v in data['edges']: adj[u].add(v);adj[v].add(u)
 rng=random.Random(0); best=None
 for trial in range(80):
  if time.monotonic()>deadline: break
  colors=[-1]*n; sat=[set() for _ in range(n)]; degree=[len(x) for x in adj]
  tie=[rng.random() if trial else -i/n for i in range(n)]
  heap=[(0,-degree[i],-tie[i],i) for i in range(n)];heapq.heapify(heap)
  while heap:
   neg,deg,_,v=heapq.heappop(heap)
   if colors[v]>=0 or -neg!=len(sat[v]):continue
   c=0
   while c in sat[v]:c+=1
   colors[v]=c
   for u in adj[v]:
    if colors[u]<0 and c not in sat[u]:
     sat[u].add(c);heapq.heappush(heap,(-len(sat[u]),-degree[u],-tie[u],u))
  k=max(colors)+1
  if best is None or k<best['objective_value']:
   best={'objective_value':k,'coloring':{str(i):colors[i] for i in range(n)}}
   if logger:logger.log_solution(k,best)
  if k<=2:break
 return best

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--instance_path');p.add_argument('--solution_path');p.add_argument('--time_limit',type=int,default=60);p.add_argument('--log_path');a=p.parse_args();start=time.monotonic();data=json.load(open(a.instance_path));logger=SolutionLogger(a.log_path,sense='minimize') if a.log_path else None
 if logger:logger.min_interval=0;logger.start_time=time.time()-(time.monotonic()-start)
 result=solve(data,start+max(0.1,a.time_limit-1),logger);result['runtime']=time.monotonic()-start
 if logger:logger.log_solution(result['objective_value'],result)
 json.dump(result,open(a.solution_path,'w'))
