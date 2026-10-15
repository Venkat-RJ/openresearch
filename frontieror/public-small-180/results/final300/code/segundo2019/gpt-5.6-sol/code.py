import argparse,json,time,random
wall_started=time.time()
from solution_logger import SolutionLogger

def configured_logger(log_path,sense):
 logger=SolutionLogger(log_path,sense=sense)
 logger.min_interval=0
 logger.start_time=wall_started
 return logger

def main():
 a=argparse.ArgumentParser(); a.add_argument('--instance_path'); a.add_argument('--solution_path'); a.add_argument('--time_limit',type=int,default=60); a.add_argument('--log_path'); a=a.parse_args(); start=time.monotonic(); end=start+max(.1,a.time_limit-1); d=json.load(open(a.instance_path)); g=d['graph']; vs=g['vertices']; adj={v:set() for v in vs}; w={}
 for u,v,z in g['edges']: adj[u].add(v); adj[v].add(u); w[min(u,v),max(u,v)]=z
 rng=random.Random(0); logger=configured_logger(a.log_path,sense='maximize') if a.log_path else None; best=-1; bestc=[]
 def weight(u,v): return w[min(u,v),max(u,v)]
 def accept(c):
  nonlocal best,bestc
  z=sum(weight(u,v) for i,u in enumerate(c) for v in c[:i])
  if z>best:
   best=z; bestc=c[:]; edges=[[min(u,v),max(u,v)] for i,u in enumerate(c) for v in c[:i]]; s={'objective_value':z,'clique_vertices':sorted(c),'clique_edges':edges}; json.dump(s,open(a.solution_path,'w')); logger and logger.log_solution(z,s)
 for u,v,z in sorted(g['edges'],key=lambda e:-e[2])[:20]: accept([u,v])
 iteration=0
 while time.monotonic()<end:
  if iteration%4 and bestc: c=[v for v in bestc if rng.random()<.65]
  else: c=[]
  candidates=set(vs)
  for v in c: candidates.intersection_update(adj[v])
  while candidates:
   if c: score=lambda v:sum(weight(v,u) for u in c)*(1+.02*len(adj[v]&candidates))
   else: score=lambda v:sum(w[min(u,v),max(u,v)] for u in adj[v])
   ranked=sorted(candidates,key=score,reverse=True); v=ranked[0] if iteration==0 else rng.choice(ranked[:min(4,len(ranked))]); c.append(v); candidates.intersection_update(adj[v])
  accept(c); iteration+=1
  if len(vs)<35 and iteration>=500: break
if __name__=='__main__': main()
